// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

import { useState, type ReactNode } from "react";
import { sendToViewer } from "../../lib/viewerBus";
import { openMolecule } from "../../lib/moleculePage";
import { covalentHandles, residuesNamed, type TargetDossier as Dossier } from "../../lib/dossier";
import type { Candidate } from "../../lib/candidates";
import type { PocketResidue, TargetFacts } from "../../lib/ttp";

interface Props {
  facts: TargetFacts;
  dossier: Dossier;
  /** The best molecule this target has, where one can be attributed to it. */
  best: Candidate | null;
  /** Put a prompt in the composer. */
  onAsk?: (text: string) => void;
}

const PREVIEW = 6;

/**
 * What the conversation learned about one target past its UniProt entry:
 * the diseases it has genetic support in, where it is expressed, the
 * pathways it sits in, its orthologs, the patents around it, the sites that
 * were characterised, the mutants that have been crystallised, the residues
 * a covalent warhead could be aimed at and the best molecule it has so far.
 *
 * Each section appears only once something has been asked, so the card
 * starts as the lookup and grows into a dossier.
 */
export function TargetDossierCard({ facts, dossier, best, onAsk }: Props) {
  const handles = covalentHandles(facts, dossier);

  const focus = (residues: PocketResidue[]) => {
    if (residues.length === 0) return;
    sendToViewer({
      action: {
        kind: "residues",
        residues: residues.filter((r) => r.resnum !== null).map((r) => ({ chain: r.chain, resnum: r.resnum as number, resname: r.resname })),
        mode: "focus",
      },
      source: "target",
    });
  };

  return (
    <div className="space-y-2">
      {best && (
        <Section label="Best molecule" hint={best.score !== undefined ? `score ${best.score.toFixed(3)}` : undefined}>
          <button
            type="button"
            onClick={() => openMolecule(best.smiles)}
            title={best.smiles}
            className="bg-sunken border-subtle hover:border-strong w-full cursor-pointer truncate rounded-md border px-2 py-1 text-left font-mono text-[10px]"
          >
            {best.name ?? best.smiles}
          </button>
        </Section>
      )}

      {dossier.diseases.length > 0 && (
        <Section label="Genetic evidence" hint="OpenTargets">
          <Rows
            items={dossier.diseases}
            render={(d) => (
              <Row key={d.disease} name={d.disease} value={[d.geneticScore !== null && `genetic ${d.geneticScore.toFixed(2)}`, d.score !== null && `overall ${d.score.toFixed(2)}`].filter(Boolean).join(" · ")} />
            )}
          />
        </Section>
      )}

      {dossier.expression.length > 0 && (
        <Section label="Expression" hint="Human Protein Atlas">
          <Rows
            items={[...dossier.expression].sort((a, b) => LEVELS.indexOf(b.level) - LEVELS.indexOf(a.level))}
            render={(e) => <Row key={`${e.tissue}/${e.cellType ?? ""}`} name={[e.tissue, e.cellType].filter(Boolean).join(" · ")} value={e.level} />}
          />
        </Section>
      )}

      {dossier.sites.length > 0 && (
        <Section label="Sites">
          {dossier.sites.map((site) => (
            <div key={site.structure ?? "site"} className="bg-sunken border-subtle w-full rounded-md border px-2 py-1.5">
              <p className="truncate text-[11px]">
                {site.siteType ?? "site"}
                <span className="text-faint ml-1 font-normal">
                  {[site.volume !== null && `${Math.round(site.volume)} Å³`, site.druggability !== null && `druggability ${site.druggability.toFixed(2)}`, site.character]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </p>
              {site.structure && <p className="text-faint truncate font-mono text-[10px]">{site.structure.split("/").pop()}</p>}
              {site.subpockets.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {site.subpockets.map((pocket) => (
                    <button
                      key={pocket.name}
                      type="button"
                      onClick={() => focus(residuesNamed(facts, pocket.residues))}
                      title={pocket.residues.join(", ")}
                      className="border-subtle hover:border-strong cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px]"
                      style={{ color: "var(--color-structure)" }}
                    >
                      {pocket.name}
                      <span className="text-faint ml-1">{pocket.residues.length}</span>
                    </button>
                  ))}
                </div>
              )}
              {site.roles.length > 0 && (
                <p className="text-muted mt-1 text-[10px]">
                  {site.roles.map((role) => `${role.residue} ${role.role}`).join(" · ")}
                </p>
              )}
            </div>
          ))}
        </Section>
      )}

      {handles.length > 0 && (
        <Section label="Covalent handles" hint="nucleophilic residues in the pocket">
          <div className="flex flex-wrap gap-1">
            {handles.slice(0, 12).map((handle) => (
              <button
                key={handle.label}
                type="button"
                onClick={() => focus(residuesNamed(facts, [handle.label]))}
                title={`${handle.residue}${handle.contacts ? ` · ${handle.contacts} contacts` : " · named by a site profile"}`}
                className="bg-sunken border-subtle hover:border-strong cursor-pointer rounded-md border px-1.5 py-px font-mono text-[10px]"
                style={{ color: "var(--color-structure)" }}
              >
                {handle.label}
                {handle.contacts > 0 && <span className="text-faint ml-1">{handle.contacts}</span>}
              </button>
            ))}
          </div>
        </Section>
      )}

      {dossier.mutants.length > 0 && (
        <Section label="Mutants" hint="as deposited">
          <Rows items={dossier.mutants} render={(m) => <Row key={`${m.pdbId}:${m.mutations}`} name={m.pdbId} value={m.mutations} hint={m.expressedIn} />} />
        </Section>
      )}

      {dossier.pathways.length > 0 && (
        <Section label="Pathways">
          <Rows items={dossier.pathways} render={(p) => <Row key={`${p.source}:${p.name}`} name={p.name} value={p.source} />} />
        </Section>
      )}

      {dossier.orthologs.length > 0 && (
        <Section label="Orthologs">
          <Rows
            items={dossier.orthologs}
            render={(o) => <Row key={o.organism} name={[o.organism, o.gene].filter(Boolean).join(" · ")} value={o.identityPct !== null ? `${Math.round(o.identityPct)}% identity` : ""} />}
          />
        </Section>
      )}

      {dossier.patents.length > 0 && (
        <Section label="Patents" hint={dossier.patentTotal !== null && dossier.patentTotal > dossier.patents.length ? `${dossier.patents.length} of ${dossier.patentTotal}` : "a flag for an attorney, not a determination"}>
          <Rows items={dossier.patents} render={(p) => <Row key={p.id} name={p.id} value={p.title ?? ""} />} />
        </Section>
      )}

      {onAsk && (
        <div className="flex">
          <button
            type="button"
            onClick={() => onAsk(ASK(facts, dossier))}
            className="text-muted hover:text-body hover:bg-inset ml-auto cursor-pointer rounded-md px-2 py-1 text-[11px] font-medium touch:min-h-11"
          >
            Fill in…
          </button>
        </div>
      )}
    </div>
  );
}

/** Expression levels worst to best, so the strongest tissues lead. */
const LEVELS = ["Not detected", "Low", "Medium", "High"];

/** What is still missing from the dossier, as a question the agent can answer. */
function ASK(facts: TargetFacts, dossier: Dossier): string {
  const name = facts.gene ?? facts.name ?? facts.uniprot ?? "the target";
  const missing = [
    dossier.diseases.length === 0 && "its genetic evidence",
    dossier.expression.length === 0 && "where it is expressed",
    dossier.pathways.length === 0 && "the pathways it sits in",
    dossier.orthologs.length === 0 && "its rat and mouse orthologs",
    dossier.sites.length === 0 && "its binding site",
    dossier.patents.length === 0 && "the patents around it",
  ].filter((part): part is string => typeof part === "string");
  if (missing.length === 0) return `Summarise what we know about ${name} and what it means for the campaign.`;
  return `For ${name}${facts.uniprot ? ` (UniProt ${facts.uniprot})` : ""}, find ${missing.join(", ")}.`;
}

function Section({ label, hint, children }: { label: string; hint?: string | false; children: ReactNode }) {
  return (
    <section className="bg-raised border-subtle shadow-card overflow-hidden rounded-xl border px-3 py-2">
      <h4 className="text-faint mb-1 text-[10px] font-semibold uppercase">
        {label}
        {hint && <span className="ml-1 font-normal normal-case">{hint}</span>}
      </h4>
      <div className="space-y-1">{children}</div>
    </section>
  );
}

/** A list that shows its first few and unfolds to the rest. */
function Rows<T>({ items, render }: { items: readonly T[]; render: (item: T) => ReactNode }) {
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, PREVIEW);
  return (
    <>
      {shown.map(render)}
      {items.length > PREVIEW && (
        <button type="button" onClick={() => setAll((v) => !v)} className="text-muted hover:text-body cursor-pointer text-[11px] font-medium">
          {all ? "Fewer" : `+${items.length - PREVIEW} more`}
        </button>
      )}
    </>
  );
}

function Row({ name, value, hint }: { name: string; value: string; hint?: string | null }) {
  return (
    <div className="flex items-baseline gap-2 text-[11px]">
      <span className="min-w-0 flex-1 truncate" title={name}>
        {name}
      </span>
      <span className="text-muted shrink-0 font-mono text-[10px]">{value}</span>
      {hint && <span className="text-faint shrink-0 truncate text-[10px]">{hint}</span>}
    </div>
  );
}
