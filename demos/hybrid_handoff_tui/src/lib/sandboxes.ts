/** The sandbox registry. Adding a sandbox = append one entry here (id, label,
 * slash, blurb, body component) — the shell, palette, and slash-command parser
 * all read from this array, so nothing else needs editing. */
import type { ComponentType } from "react";
import { EmailBody } from "../components/sandboxes/EmailBody";
import { GenericBody } from "../components/sandboxes/GenericBody";
import type { BodyProps } from "./snapshot";

export interface SandboxDef {
  id: string;
  label: string;
  slash: string;
  blurb: string;
  body: ComponentType<BodyProps>;
}

export const SANDBOXES: SandboxDef[] = [
  {
    id: "email",
    label: "EMAIL TRIAGE AGENT",
    slash: "/email",
    blurb: "Triage an inbox — search, read, cross-check calendar + docs",
    body: EmailBody,
  },
  {
    id: "calendar",
    label: "CALENDAR AGENT",
    slash: "/calendar",
    blurb: "Find a conflict-free meeting slot and book it",
    body: GenericBody,
  },
  {
    id: "expense",
    label: "EXPENSE AGENT",
    slash: "/expense",
    blurb: "Check a receipt against policy and submit if compliant",
    body: GenericBody,
  },
  {
    id: "incident",
    label: "INCIDENT AGENT",
    slash: "/incident",
    blurb: "Triage an alert from logs + runbook, recommend an action",
    body: GenericBody,
  },
];

export function getSandbox(id: string): SandboxDef {
  return SANDBOXES.find((s) => s.id === id) ?? SANDBOXES[0];
}

export function sandboxBySlash(slash: string): SandboxDef | undefined {
  return SANDBOXES.find((s) => s.slash === slash);
}
