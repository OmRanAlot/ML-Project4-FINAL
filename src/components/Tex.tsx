import katex from "katex";
import "katex/dist/katex.min.css";
import { useMemo } from "react";

/** Renders a static, author-written TeX string (never user input). */
export function Tex({ math, display = false }: { math: string; display?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(math, { displayMode: display, throwOnError: false, trust: false }),
    [math, display],
  );
  return <span className={display ? "tex tex--display" : "tex"} dangerouslySetInnerHTML={{ __html: html }} />;
}
