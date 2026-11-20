import { AlertTriangle } from 'lucide-react'

export function Warnings({ items = [] }: { items?: string[] }) {
  if (!items.length) return null
  return <div className="warnings-panel"><div className="warning-heading"><AlertTriangle size={17} /><strong>Assessment notes</strong></div><ul>{items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}</ul></div>
}
