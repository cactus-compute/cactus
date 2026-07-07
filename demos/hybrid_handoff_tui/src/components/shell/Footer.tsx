import type { InputRenderable } from "@opentui/core";
import { useRef } from "react";
import { COLORS } from "../../theme";

export function Footer({
  sandboxId,
  disabled,
  statusLine,
  onCommand,
}: {
  sandboxId: string;
  disabled: boolean;
  statusLine: string;
  onCommand: (text: string) => void;
}) {
  const ref = useRef<InputRenderable>(null);
  return (
    <box
      style={{
        flexDirection: "column",
        flexShrink: 0,
        backgroundColor: COLORS.panelAlt,
        border: ["top"],
        borderColor: COLORS.border,
        paddingX: 1,
      }}
    >
      <text fg={COLORS.dim}>{statusLine || " "}</text>
      <box style={{ flexDirection: "row" }}>
        <text fg={COLORS.local}>{`cactus:~/${sandboxId} $ `}</text>
        <input
          ref={ref}
          focused={!disabled}
          placeholder="type a prompt, or /email /expense /calendar /run /reset · ⌃K for palette"
          onSubmit={(value) => {
            const text = (value || "").trim();
            if (!text) return;
            onCommand(text);
            if (ref.current) ref.current.value = "";
          }}
          style={{ flexGrow: 1 }}
        />
      </box>
    </box>
  );
}
