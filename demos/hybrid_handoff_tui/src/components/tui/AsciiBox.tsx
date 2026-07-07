import type { ReactNode } from "react";
import { COLORS } from "../../theme";

export function AsciiBox({
  title,
  color = COLORS.border,
  fill = true,
  style,
  children,
}: {
  title?: string;
  color?: string;
  fill?: boolean;
  style?: Record<string, unknown>;
  children?: ReactNode;
}) {
  return (
    <box
      title={title ? ` ${title} ` : undefined}
      titleColor={color}
      style={{
        flexDirection: "column",
        border: true,
        borderColor: color,
        backgroundColor: fill ? COLORS.panel : COLORS.bg,
        paddingX: 1,
        ...style,
      }}
    >
      {children}
    </box>
  );
}
