// Captures the comparison playground (flightHelmet, view 2 = "Leather strap and buckle" by default) for a list of
// methods: a plain screenshot, a diff-vs-original screenshot and the playground's compareStats (MAE, PSNR, ...).
//
// Every GPU job must hold the shared lock, so run it through the lock script:
//   powershell -File files\gpulock.ps1 node tools\quality\capture.mjs --out output\quality\shots \
//     --entry base=runtime:v2best --entry ktx=ktx --entry load=loadtime:v2best
//
// An entry is label=mode[:suffix]. Modes: original, ktx, runtime, loadtime (neural, <suffix> as in --neural-suffix).
//
// Environment: BABYLON_NEURAL_DIR (folder with node_modules/@playwright/test and puppeteer, default
// C:/Users/cedric/dev/babylon-neural), PG_URL (default http://localhost:1338/).
import { createRequire } from "node:module";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const opt = { entry: [] };
for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (!a.startsWith("--")) throw new Error("unexpected argument " + a);
    const k = a.slice(2);
    const v = args[++i];
    if (k === "entry") opt.entry.push(v);
    else opt[k] = v;
}
const engine = opt.engine ?? "webgl";
const asset = opt.asset ?? "flightHelmet";
const view = Number(opt.view ?? 2);
const outDir = path.resolve(opt.out ?? "output/quality/shots");
const neuralBase = opt["neural-base"] ?? "http://localhost:8081/output/quality/";
const originalBase = opt["original-base"] ?? "http://localhost:8081/assets/";
const pgId = opt.id ?? "PV2IKD";
const pgRev = opt.rev ?? "4";
const PG = process.env.PG_URL ?? "http://localhost:1338/";
const nnDir = process.env.BABYLON_NEURAL_DIR ?? "C:/Users/cedric/dev/babylon-neural";
if (!opt.entry.length) throw new Error("no --entry label=mode[:suffix]");

const require = createRequire(path.join(nnDir, "package.json"));
const { chromium } = require("@playwright/test");
const { default: puppeteer } = await import(pathToFileURL(path.join(nnDir, "node_modules/puppeteer/lib/puppeteer/puppeteer.js")).href);
fs.mkdirSync(outDir, { recursive: true });
const ctx = await chromium.launchPersistentContext(path.join(outDir, ".profile-" + engine), {
    headless: true,
    executablePath: await puppeteer.executablePath(),
    viewport: { width: 1920, height: 1000 },
    deviceScaleFactor: 1,
    args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"],
});
const page = await ctx.newPage();
const errors = [];
page.on("console", (m) => m.type() === "error" && errors.push(m.text()));
page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
const q = new URLSearchParams({ neuralBase, originalBase, asset, view: String(view), mode: "original" });
if (engine === "webgpu") q.set("webgpu", "");
await page.goto(`${PG}?${q.toString().replace(/webgpu=(&|$)/, "webgpu$1")}#${pgId}#${pgRev}`);
await page.waitForFunction(() => !!window.__neuralCompare, null, { timeout: 180000 });
await page.evaluate(() => window.__neuralCompare.ready());

const kindOf = { runtime: "neural", loadtime: "neural" };
const resultsPath = path.join(outDir, `captures_${engine}_${asset}_v${view}.json`);
const results = fs.existsSync(resultsPath) ? JSON.parse(fs.readFileSync(resultsPath, "utf8")) : {};
for (const e of opt.entry) {
    const [label, spec] = e.split("=");
    const [mode, suffix] = spec.split(":");
    const row = { label, mode, suffix: suffix ?? null, engine, asset, view };
    for (const cmp of [false, true]) {
        const e0 = errors.length;
        const s = await page.evaluate(
            async ([a, m, v, cmp, kind, suffix]) => {
                const n = window.__neuralCompare;
                if (kind && suffix) await n.setFile(kind, suffix);
                n.state.asset = a;
                n.state.mode = m;
                n.state.view = v;
                n.state.compare = cmp;
                await n.setAsset(a);
                await n.ready();
                return JSON.parse(JSON.stringify(n.stats(), (k, x) => (x === Infinity ? "inf" : x)));
            },
            [asset, mode, view, cmp, kindOf[mode] ?? null, suffix ?? null]
        );
        await page.waitForTimeout(400);
        const shot = path.join(outDir, `${engine}_${asset}_v${view}_${label}_${cmp ? "cmp" : "plain"}.png`);
        await page.locator("#renderCanvas").screenshot({ path: shot });
        row[cmp ? "cmpShot" : "plainShot"] = shot;
        row.urls = s.urls;
        row.loadMs = s.loadMs;
        if (cmp) row.compareStats = s.compareStats;
        row.errors = (row.errors ?? []).concat(errors.slice(e0));
    }
    results[label] = row;
    const cs = row.compareStats ?? {};
    console.log(`${label.padEnd(16)} ${spec.padEnd(24)} MAE ${cs.mae?.toFixed?.(3)} MAEmesh ${cs.maeMesh?.toFixed?.(3)} PSNR ${cs.psnr?.toFixed?.(2)} ${row.errors.length ? "ERR " + row.errors[0].slice(0, 120) : ""}`);
}
fs.writeFileSync(resultsPath, JSON.stringify(results, null, 1));
await ctx.close();
process.exit(errors.length && opt.strict ? 1 : 0);
