declare module 'lucide-react' {
  import type { ReactElement, SVGProps } from 'react'

  export type LucideIcon = (props: SVGProps<SVGSVGElement> & {
    size?: number | string
    strokeWidth?: number | string
  }) => ReactElement

  export const Activity: LucideIcon
  export const AlertTriangle: LucideIcon
  export const ArrowLeft: LucideIcon
  export const ArrowRight: LucideIcon
  export const ArrowUpRight: LucideIcon
  export const Beaker: LucideIcon
  export const Bell: LucideIcon
  export const BookOpen: LucideIcon
  export const Braces: LucideIcon
  export const Camera: LucideIcon
  export const Check: LucideIcon
  export const CheckCircle2: LucideIcon
  export const ChevronDown: LucideIcon
  export const CircleAlert: LucideIcon
  export const CircleHelp: LucideIcon
  export const CircleX: LucideIcon
  export const Clock3: LucideIcon
  export const ClipboardList: LucideIcon
  export const Dumbbell: LucideIcon
  export const FileSearch: LucideIcon
  export const FileClock: LucideIcon
  export const FileVideo2: LucideIcon
  export const Gauge: LucideIcon
  export const Info: LucideIcon
  export const LayoutDashboard: LucideIcon
  export const ListChecks: LucideIcon
  export const LockKeyhole: LucideIcon
  export const LoaderCircle: LucideIcon
  export const LogOut: LucideIcon
  export const Menu: LucideIcon
  export const Ruler: LucideIcon
  export const ScanLine: LucideIcon
  export const Server: LucideIcon
  export const Settings2: LucideIcon
  export const ShieldCheck: LucideIcon
  export const Sparkles: LucideIcon
  export const Target: LucideIcon
  export const TrendingUp: LucideIcon
  export const UploadCloud: LucideIcon
  export const Video: LucideIcon
  export const X: LucideIcon
}
