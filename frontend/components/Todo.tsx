// Visible placeholder so teammates can see which Jira task fills each part of a page.
export default function Todo({ task, children }: { task: string; children?: React.ReactNode }) {
  return (
    <div className="rounded border border-dashed border-amber-400 bg-amber-50 p-3 text-sm text-amber-900">
      <strong>TODO ({task})</strong>
      {children && <div className="mt-1">{children}</div>}
    </div>
  );
}
