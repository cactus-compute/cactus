import { COLORS } from "../../theme";
import type { BodyProps } from "../../lib/snapshot";
import { senderName, truncate } from "../../util";
import { Markdown } from "../tui/Markdown";

function readEmailIds(turns: BodyProps["completedTurns"]): Set<string> {
  const ids = new Set<string>();
  for (const t of turns) {
    if (t.tool_name === "read_email") {
      const id = t.tool_args?.id;
      if (typeof id === "string") ids.add(id);
    }
  }
  return ids;
}

export function EmailBody({ completedTurns, finalText, inbox }: BodyProps) {
  const read = readEmailIds(completedTurns);
  const activeId = [...read].pop() ?? null;
  return (
    <scrollbox
      stickyScroll
      stickyStart="top"
      scrollY
      style={{ flexGrow: 1, paddingX: 1, paddingTop: 1, backgroundColor: COLORS.bg }}
    >
      <box style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 1 }}>
        <text fg={COLORS.local}>{">> INBOX · TRIAGE"}</text>
        <text fg={COLORS.dim}>{`${read.size}/${inbox.length} READ`}</text>
      </box>

      {inbox.map((m) => {
        const isRead = read.has(m.id);
        const isActive = m.id === activeId;
        const mark = isRead ? "✓" : m.unread ? "●" : " ";
        const markColor = isRead ? COLORS.local : m.unread ? COLORS.threshold : COLORS.faint;
        return (
          <box
            key={m.id}
            style={{
              flexDirection: "row",
              paddingX: 1,
              backgroundColor: isActive ? COLORS.panel : COLORS.bg,
            }}
          >
            <text fg={COLORS.local}>{isActive ? "▎ " : "  "}</text>
            <text fg={markColor}>{`${mark} `}</text>
            <box style={{ flexDirection: "column" }}>
              <text fg={isActive ? COLORS.local : isRead ? COLORS.dim : COLORS.text}>
                {truncate(m.subject, 46)}
              </text>
              <text fg={COLORS.faint}>{truncate(senderName(m.from), 46)}</text>
            </box>
          </box>
        );
      })}

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
