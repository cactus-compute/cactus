import { COLORS } from "../../theme";
import type { BodyProps } from "../../lib/snapshot";
import { Markdown } from "../tui/Markdown";

export function GenericBody({ finalText }: BodyProps) {
  return (
    <scrollbox scrollY style={{ flexGrow: 1, paddingX: 2, paddingTop: 2, backgroundColor: COLORS.bg }}>
      <text fg={COLORS.dim}>This sandbox is wired to the engine — its body view is not built yet.</text>
      <text fg={COLORS.faint}>The trace panel on the right shows live local↔cloud routing for it.</text>
      {finalText ? (
        <box
          title=" ✦ AGENT RESULT "
          titleColor={COLORS.local}
          style={{
            flexDirection: "column",
            border: true,
            borderColor: COLORS.local,
            backgroundColor: COLORS.panel,
            marginTop: 1,
            paddingX: 1,
          }}
        >
          <Markdown content={finalText} />
        </box>
      ) : null}
    </scrollbox>
  );
}
