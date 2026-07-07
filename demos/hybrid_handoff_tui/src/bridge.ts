/** Spawns the resident Python bridge and turns its NDJSON stdout into typed
 * events. The bridge is the brain (it owns the Cactus FFI, where probe
 * confidence and cloud token usage live); this client only sends commands and
 * renders events. */
import { join, resolve } from "node:path";
import type { BridgeEvent, Command } from "./types";

const HERE = import.meta.dir;
const REPO_ROOT = resolve(HERE, "..", "..", "..");
const PYTHON = join(REPO_ROOT, "venv", "bin", "python");
const SCRIPT = join(REPO_ROOT, "demos", "hybrid_handoff", "bridge.py");

export class Bridge {
  private proc: ReturnType<typeof Bun.spawn>;
  private buf = "";

  constructor(
    private onEvent: (event: BridgeEvent) => void,
    private onStderr?: (line: string) => void,
  ) {
    this.proc = Bun.spawn([PYTHON, SCRIPT, "--sandbox", "email"], {
      cwd: REPO_ROOT,
      stdin: "pipe",
      stdout: "pipe",
      stderr: "pipe",
    });
    void this.pump(this.proc.stdout as ReadableStream<Uint8Array>, (line) => {
      try {
        this.onEvent(JSON.parse(line) as BridgeEvent);
      } catch {
        /* ignore non-JSON noise */
      }
    });
    if (this.onStderr) {
      void this.pump(this.proc.stderr as ReadableStream<Uint8Array>, (line) =>
        this.onStderr?.(line),
      );
    }
  }

  private async pump(stream: ReadableStream<Uint8Array>, onLine: (line: string) => void) {
    const dec = new TextDecoder();
    let acc = "";
    for await (const chunk of stream) {
      acc += dec.decode(chunk, { stream: true });
      let nl: number;
      while ((nl = acc.indexOf("\n")) >= 0) {
        const line = acc.slice(0, nl).trim();
        acc = acc.slice(nl + 1);
        if (line) onLine(line);
      }
    }
  }

  send(cmd: Command): void {
    this.proc.stdin.write(JSON.stringify(cmd) + "\n");
    this.proc.stdin.flush();
  }

  quit(): void {
    try {
      this.send({ type: "quit" });
    } catch {
      /* already gone */
    }
    try {
      this.proc.kill();
    } catch {
      /* already gone */
    }
  }
}
