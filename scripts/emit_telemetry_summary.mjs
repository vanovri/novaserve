#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import zlib from 'node:zlib';
import { execSync } from 'node:child_process';

function parseArgs(argv) {
  const args = {
    report: 'docs/benchmarks/serving_sla.md',
    progress: 'PROGRESS.md',
    verifyLog: null,
    out: 'studio/public/telemetry/summary.json',
    strict: false,
    allowDirty: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === '--report') args.report = argv[++i];
    else if (flag === '--progress') args.progress = argv[++i];
    else if (flag === '--verify-log') args.verifyLog = argv[++i];
    else if (flag === '--out') args.out = argv[++i];
    else if (flag === '--strict') args.strict = true;
    else if (flag === '--allow-dirty') args.allowDirty = true;
    else throw new Error(`Unknown argument: ${flag}`);
  }
  return args;
}

function normalizeNumber(raw) {
  const s = String(raw).trim().replace(/_/g, '');
  if (/^\d{1,3}(,\d{3})+$/.test(s)) return Number(s.replace(/,/g, ''));
  const num = Number(s.replace(/,/g, ''));
  if (!Number.isFinite(num)) {
    throw new Error(`Cannot parse numeric token "${raw}"`);
  }
  return num;
}

function matchFirst(corpus, patterns) {
  for (const regex of patterns) {
    const m = corpus.match(regex);
    if (m) return m;
  }
  return null;
}

function extractNumber(corpus, label, patterns, errors, required = true) {
  const m = matchFirst(corpus, patterns);
  if (!m || m[1] === undefined) {
    if (required) {
      errors.push(`Missing required metric [${label}] in benchmark report/logs`);
    }
    return null;
  }
  return normalizeNumber(m[1]);
}

function inspectWasmFile(relPath) {
  const buf = fs.readFileSync(relPath);
  if (buf.length === 0) return null;

  // Validate WebAssembly magic header (\0asm)
  if (
    buf.length < 4 ||
    buf[0] !== 0x00 ||
    buf[1] !== 0x61 ||
    buf[2] !== 0x73 ||
    buf[3] !== 0x6d
  ) {
    throw new Error(`Invalid WASM magic bytes in ${relPath}`);
  }

  const gzipped = zlib.gzipSync(buf, { level: 9 });
  const sha256 = crypto.createHash('sha256').update(buf).digest('hex');

  return {
    relPath,
    bytes: buf.length,
    kb: Number((buf.length / 1024).toFixed(2)),
    kbRounded: Math.round(buf.length / 1024),
    siKb: Number((buf.length / 1000).toFixed(2)),
    gzipBytes: gzipped.length,
    gzipKb: Number((gzipped.length / 1024).toFixed(2)),
    sha256,
  };
}

function findNovaServeWasmBinary() {
  const exactCandidates = [
    'studio/public/novaserve_wasm.wasm',
    'studio/dist/novaserve_wasm.wasm',
    'target/wasm32-unknown-unknown/release/novaserve_wasm.wasm',
  ];

  for (const rel of exactCandidates) {
    if (fs.existsSync(rel)) {
      const stat = fs.statSync(rel);
      if (stat.isFile() && stat.size > 0) {
        return inspectWasmFile(rel);
      }
    }
  }
  return null;
}

function countWorkspaceCrates() {
  const cratesDir = 'crates';
  if (!fs.existsSync(cratesDir)) return 0;
  return fs
    .readdirSync(cratesDir, { withFileTypes: true })
    .filter((d) => d.isDirectory() && fs.existsSync(path.join(cratesDir, d.name, 'Cargo.toml')))
    .length;
}

function getGitMeta() {
  const commitSha = execSync('git rev-parse --short HEAD', { encoding: 'utf8' }).trim();
  const statusOut = execSync('git status --porcelain', { encoding: 'utf8' }).trim();
  return { commitSha, workingTreeClean: statusOut.length === 0 };
}

function assertLogFreshness(verifyLogPath, currentSha, errors) {
  const shaFile = verifyLogPath.replace(/\.log$/, '.sha');
  if (!fs.existsSync(shaFile)) return;
  const recordedSha = fs.readFileSync(shaFile, 'utf8').trim();
  if (recordedSha && recordedSha !== currentSha) {
    errors.push(
      `Stale verify-all log: recorded at commit ${recordedSha}, current HEAD is ${currentSha}. Re-run "make verify-live".`
    );
  }
}

function main() {
  const args = parseArgs(process.argv);
  const errors = [];
  const inputs = [];

  if (args.report && fs.existsSync(args.report)) {
    inputs.push({ name: args.report, content: fs.readFileSync(args.report, 'utf8') });
  } else {
    errors.push(`Benchmark report not found: ${args.report}`);
  }

  let progressText = '';
  if (args.progress && fs.existsSync(args.progress)) {
    progressText = fs.readFileSync(args.progress, 'utf8');
    inputs.push({ name: args.progress, content: progressText });
  }

  if (args.verifyLog && fs.existsSync(args.verifyLog)) {
    inputs.push({ name: args.verifyLog, content: fs.readFileSync(args.verifyLog, 'utf8') });
  }

  if (inputs.length === 0) {
    console.error('[telemetry] STRICT GATE FAILED: no input report or log files found.');
    process.exit(1);
  }

  const reportCorpus = (inputs[0]?.content ?? '').replace(/\x1B\[[0-9;]*[A-Za-z]/g, '');
  const fullCorpus = inputs
    .map((x) => x.content)
    .join('\n')
    .replace(/\x1B\[[0-9;]*[A-Za-z]/g, '');

  const reportSha256 = crypto.createHash('sha256').update(reportCorpus).digest('hex');
  const { commitSha, workingTreeClean } = getGitMeta();

  if (args.verifyLog && fs.existsSync(args.verifyLog)) {
    assertLogFreshness(args.verifyLog, commitSha, errors);
  }

  let wasmInfo = null;
  try {
    wasmInfo = findNovaServeWasmBinary();
    if (!wasmInfo) {
      errors.push('Compiled novaserve-wasm binary (novaserve_wasm.wasm) not found on disk');
    }
  } catch (err) {
    errors.push(err instanceof Error ? err.message : String(err));
  }

  const cratesCount = countWorkspaceCrates();
  if (cratesCount === 0) {
    errors.push('No crates/*/Cargo.toml found in workspace');
  }

  // --- 1. Canonical Serving SLA Report (`docs/benchmarks/serving_sla.md`) ---
  const backendMatch = reportCorpus.match(/Active SIMD Kernel Backend:\*\*\s*`([^`]+)`/i);
  const simdBackend = backendMatch ? backendMatch[1].trim() : null;
  if (!simdBackend) {
    errors.push('Missing Active SIMD Kernel Backend in serving_sla.md');
  }

  const fp32Row = reportCorpus.match(
    /\|\s*`FP32`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/i
  );
  const q8Row = reportCorpus.match(
    /\|\s*`Q8_0`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/i
  );
  const q4Row = reportCorpus.match(
    /\|\s*`Q4_0`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)x\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|/i
  );

  const fp32Batch16TokPerSec = fp32Row ? normalizeNumber(fp32Row[5]) : null;
  const q8CompressionRatio = q8Row ? normalizeNumber(q8Row[2]) : null;
  const q8SingleTokPerSec = q8Row ? normalizeNumber(q8Row[3]) : null;
  const q4CompressionRatio = q4Row ? normalizeNumber(q4Row[2]) : null;
  if (q8CompressionRatio === null) errors.push('Missing Q8_0 quantization row in serving_sla.md');
  if (q4CompressionRatio === null) errors.push('Missing Q4_0 quantization row in serving_sla.md');
  if (fp32Batch16TokPerSec === null) errors.push('Missing FP32 quantization row in serving_sla.md');

  const contBatchTokPerSec = extractNumber(
    reportCorpus,
    'sla.continuous_batch_tok_s',
    [/\|\s*Continuous Batching\s*\(32 seqs\)\s*\|\s*32\s*\|\s*([\d.]+)\s*\|/i],
    errors
  );

  const seqBatchTokPerSec = extractNumber(
    reportCorpus,
    'sla.sequential_batch_tok_s',
    [/\|\s*Sequential Locking\s*\(batch=1\)\s*\|\s*32\s*\|\s*([\d.]+)\s*\|/i],
    errors
  );

  const chunkedTailMatch = reportCorpus.match(
    /Unchunked\s*`([\d.]+)\s*ms`\s*->\s*Chunked\s*\(`?B=64`?\)\s*`([\d.]+)\s*ms`\s*\(\*\*`([\d.]+)x`\s*tail-spike reduction\*\*\)/i
  );
  const unchunkedItlP99Ms = chunkedTailMatch ? normalizeNumber(chunkedTailMatch[1]) : null;
  const chunkedItlP99Ms = chunkedTailMatch ? normalizeNumber(chunkedTailMatch[2]) : null;
  const chunkedItlDropRatio = chunkedTailMatch ? normalizeNumber(chunkedTailMatch[3]) : null;
  if (chunkedItlDropRatio === null) {
    errors.push('Missing Chunked Prefill ITL p99 tail-spike reduction line in serving_sla.md');
  }

  const radixOffMatch = reportCorpus.match(
    /\|\s*`Radix OFF`\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%\s*\|/i
  );
  const radixOnMatch = reportCorpus.match(
    /\|\s*`Radix ON`\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*\*\*([\d.]+)%\*\*\s*\|/i
  );
  const radixSpeedupMatch = reportCorpus.match(
    /Radix\s*`TTFT p50`\s*Speedup:\*\*\s*\*\*`([\d.]+)x`\s*faster\*\*\s*\(`([\d.]+)x`\s*end-to-end wall speedup\)/i
  );

  const radixOffTtftP50Ms = radixOffMatch ? normalizeNumber(radixOffMatch[2]) : null;
  const radixOnTtftP50Ms = radixOnMatch ? normalizeNumber(radixOnMatch[2]) : null;
  const radixHitRatePct = radixOnMatch ? normalizeNumber(radixOnMatch[5]) : null;
  const radixTtftSpeedup = radixSpeedupMatch ? normalizeNumber(radixSpeedupMatch[1]) : null;
  const radixWallSpeedup = radixSpeedupMatch ? normalizeNumber(radixSpeedupMatch[2]) : null;
  if (radixTtftSpeedup === null || radixHitRatePct === null) {
    errors.push('Missing Radix Prefix Cache TTFT speedup / hit rate in serving_sla.md');
  }

  const specRowMatch = reportCorpus.match(
    /\|\s*Speculative\s*\(`Draft-500K`\s*\+\s*`Tiny-16M`\)\s*\|\s*(\d+)\s*\|\s*\*\*([\d.]+)%\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)\*\*\s*\|\s*\*\*([\d.]+)x\*\*\s*\|/i
  );
  const speculativeK = specRowMatch ? normalizeNumber(specRowMatch[1]) : null;
  const speculativeAlphaPct = specRowMatch ? normalizeNumber(specRowMatch[2]) : null;
  const speculativeMeanTokPerPass = specRowMatch ? normalizeNumber(specRowMatch[3]) : null;
  const speculativeTokPerSec = specRowMatch ? normalizeNumber(specRowMatch[4]) : null;
  const speculativeTpotMs = specRowMatch ? normalizeNumber(specRowMatch[5]) : null;
  const speculativeTpotSpeedup = specRowMatch ? normalizeNumber(specRowMatch[6]) : null;
  if (speculativeTpotSpeedup === null || speculativeAlphaPct === null) {
    errors.push('Missing Speculative Decoding K=4 row in serving_sla.md');
  }

  // --- 2. Oracle, Safety & Verification Proof from PROGRESS.md ---
  const fp32GemmGflops = extractNumber(
    fullCorpus,
    'tensor.fp32_gemm_gflops',
    [/`FP32`\s*GEMM\s*`([\d.]+)\s*GFLOPS`/i],
    errors,
    false
  );
  const miriTestsMatch = fullCorpus.match(/Miri\s*\((\d+)\/(\d+),\s*(\d+)\s*UB\)/i);
  const miriPassed = miriTestsMatch ? normalizeNumber(miriTestsMatch[1]) : null;
  const miriUbCount = miriTestsMatch ? normalizeNumber(miriTestsMatch[3]) : null;

  const jsonDfaValid = extractNumber(
    fullCorpus,
    'frontend.json_dfa_valid',
    [/`([\d,]+)\/5,000`\s*\(`100%`\)\s*valid JSON schemas/i],
    errors,
    false
  );
  const kvExtFragPct = extractNumber(
    fullCorpus,
    'kv.ext_fragmentation_pct',
    [/`([\d.]+)%`\s*ext fragmentation/i],
    errors,
    false
  );
  const pagedMaxAbsDiffMatch = fullCorpus.match(/`max_abs_diff\s*=\s*([0-9.eE+-]+)`/i);
  const pagedMaxAbsDiff = pagedMaxAbsDiffMatch ? Number(pagedMaxAbsDiffMatch[1]) : null;

  const chaosMaxLogprobMatch = fullCorpus.match(/`max_logprob_diff\s*=\s*([0-9.eE+-]+)`/i);
  const chaosMaxLogprobDiff = chaosMaxLogprobMatch ? Number(chaosMaxLogprobMatch[1]) : null;

  const tvdRangeMatch = fullCorpus.match(/`TVD`\s*=\s*`([\d.]+)\.\.([\d.]+)`/i);
  const speculativeTvdMin = tvdRangeMatch ? Number(tvdRangeMatch[1]) : null;
  const speculativeTvdMax = tvdRangeMatch ? Number(tvdRangeMatch[2]) : null;

  const chaosScenarios = extractNumber(
    fullCorpus,
    'oracle.chaos_scenarios',
    [/([\d,]+)\s*Chaos Batch Scenarios/i],
    errors,
    false
  );
  const wasmE2eMatch = fullCorpus.match(/`(\d+)\/(\d+)`\s*WASM Studio E2E/i);
  const wasmE2ePassed = wasmE2eMatch ? normalizeNumber(wasmE2eMatch[1]) : null;

  // Hard Safety & Statistical Anti-Cheat Assertions
  if (miriUbCount !== null && miriUbCount !== 0) {
    errors.push(`Miri UB Violation: ub_count=${miriUbCount}`);
  }
  if (kvExtFragPct !== null && kvExtFragPct !== 0) {
    errors.push(`KV-Cache Fragmentation Violation: kvExtFragPct=${kvExtFragPct}`);
  }
  if (speculativeTvdMax !== null && speculativeTvdMax >= 0.015) {
    errors.push(`Speculative TVD Violation: max TVD=${speculativeTvdMax} >= 0.015`);
  }
  if (pagedMaxAbsDiff !== null && pagedMaxAbsDiff >= 1e-5) {
    errors.push(`PagedAttention Parity Violation: max_abs_diff=${pagedMaxAbsDiff} >= 1e-5`);
  }

  if (errors.length > 0 && args.strict) {
    console.error('[telemetry] STRICT GATE FAILED — refusing to write summary.json:');
    for (const err of errors) {
      console.error(`  - ${err}`);
    }
    process.exit(1);
  }

  const status = errors.length === 0 ? 'VERIFIED_LIVE' : 'PARTIAL_VERIFY';

  const summary = {
    schemaVersion: '1.0.0',
    slug: 'novaserve',
    status,
    commitSha,
    workingTreeClean,
    verifiedAt: new Date().toISOString(),
    wasmAssetPath: '/labs/novaserve/assets/novaserve_wasm.wasm',
    wasmSizeKb: wasmInfo ? wasmInfo.kb : null,
    wasmSizeBytes: wasmInfo ? wasmInfo.bytes : null,
    wasmGzipKb: wasmInfo ? wasmInfo.gzipKb : null,
    cratesCount,
    headlineMetrics: [
      {
        id: 'radix_ttft_speedup',
        label: 'Radix Prefix TTFT',
        value:
          radixTtftSpeedup !== null && radixHitRatePct !== null
            ? `${radixTtftSpeedup.toFixed(2)}x (${radixHitRatePct.toFixed(1)}% hit)`
            : null,
      },
      {
        id: 'chunked_itl_p99',
        label: 'Chunked Prefill ITL p99',
        value:
          chunkedItlDropRatio !== null && chunkedItlP99Ms !== null
            ? `${chunkedItlDropRatio.toFixed(2)}x drop (${chunkedItlP99Ms.toFixed(2)} ms)`
            : null,
      },
      {
        id: 'speculative_tpot',
        label: 'Speculative K=4 TPOT',
        value:
          speculativeTpotSpeedup !== null && q8CompressionRatio !== null
            ? `${speculativeTpotSpeedup.toFixed(2)}x (Q8 ${q8CompressionRatio.toFixed(2)}x)`
            : null,
      },
    ],
    extendedProof: {
      simdBackend,
      fp32GemmGflops,
      fp32Batch16TokPerSec,
      q8CompressionRatio,
      q8SingleTokPerSec,
      q4CompressionRatio,
      seqBatch32TokPerSec: seqBatchTokPerSec,
      contBatch32TokPerSec: contBatchTokPerSec,
      unchunkedItlP99Ms,
      chunkedItlP99Ms,
      chunkedItlDropRatio,
      radixOffTtftP50Ms,
      radixOnTtftP50Ms,
      radixHitRatePct,
      radixTtftSpeedup,
      radixWallSpeedup,
      speculativeK,
      speculativeAlphaPct,
      speculativeMeanTokPerPass,
      speculativeTokPerSec,
      speculativeTpotMs,
      speculativeTpotSpeedup,
      speculativeTvdMin,
      speculativeTvdMax,
      pagedMaxAbsDiff,
      chaosScenarios,
      chaosMaxLogprobDiff,
      jsonDfaValid,
      kvExtFragPct,
      miriPassed,
      miriUbCount,
      wasmE2ePassed,
      wasmGzipBytes: wasmInfo ? wasmInfo.gzipBytes : null,
    },
    provenance: {
      reportSha256,
      sourceFiles: inputs.map((x) => x.name),
      wasmSourcePath: wasmInfo ? wasmInfo.relPath : null,
      wasmSha256: wasmInfo ? wasmInfo.sha256 : null,
      missingOrWarnings: errors,
    },
  };

  // Atomic write (.tmp -> rename)
  const outDir = path.dirname(args.out);
  fs.mkdirSync(outDir, { recursive: true });
  const tmpPath = `${args.out}.tmp.${process.pid}`;
  fs.writeFileSync(tmpPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8');
  fs.renameSync(tmpPath, args.out);

  console.log(
    `[telemetry] Wrote ${args.out} (slug=novaserve, status=${status}, commit=${commitSha}, wasm=${summary.wasmSizeKb} KB / ${summary.wasmGzipKb} KB gzip, sha256=${reportSha256.slice(0, 12)})`
  );
}

main();
