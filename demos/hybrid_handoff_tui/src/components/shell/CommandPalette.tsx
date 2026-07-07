import type { SelectOption } from "@opentui/core";
import { COLORS } from "../../theme";
import { SANDBOXES } from "../../lib/sandboxes";

const OPTIONS: SelectOption[] = [
  ...SANDBOXES.map((s) => ({ name: `${s.slash}   ${s.label}`, description: s.blurb, value: s.slash })),
  { name: "/run    run default task", description: "Run this sandbox's example task", value: "/run" },
  { name: "/reset  clear the run", description: "Clear the current trace and result", value: "/reset" },
];

export function CommandPalette({ onPick }: { onPick: (value: string) => void }) {
  return (
    <box
      title=" ⌃K  COMMAND PALETTE "
      titleColor={COLORS.local}
      style={{
        position: "absolute",
        top: 5,
        left: 6,
        right: 6,
        zIndex: 100,
        flexDirection: "column",
        border: true,
        borderColor: COLORS.local,
        backgroundColor: COLORS.panel,
        paddingX: 1,
      }}
    >
      <select
        options={OPTIONS}
        focused
        showDescription
        wrapSelection
        descriptionColor={COLORS.faint}
        textColor={COLORS.text}
        backgroundColor={COLORS.panel}
        focusedBackgroundColor={COLORS.local}
        focusedTextColor={COLORS.ink}
        onSelect={(_index, option) => {
          if (option?.value) onPick(String(option.value));
        }}
        style={{ height: 12 }}
      />
      <text fg={COLORS.faint}>{" ↑↓ select · Enter run · Esc close"}</text>
    </box>
  );
}
