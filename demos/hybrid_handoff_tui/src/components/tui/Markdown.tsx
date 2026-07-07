import { SyntaxStyle } from "@opentui/core";
import { COLORS } from "../../theme";

let cached: SyntaxStyle | null = null;

function style(): SyntaxStyle {
  if (!cached) cached = SyntaxStyle.create();
  return cached;
}

export function Markdown({ content, fg = COLORS.text }: { content: string; fg?: string }) {
  return <markdown content={content} syntaxStyle={style()} conceal fg={fg} />;
}
