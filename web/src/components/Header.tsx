import { Bell, ChevronDown, LogOut, Menu } from 'lucide-react'

interface HeaderProps {
  title: string
  eyebrow?: string
  onMenuClick?: () => void
  admin?: boolean
  displayName?: string
  onLogout?: () => void
}

export function Header({ title, eyebrow = 'ATHLETE WORKSPACE', onMenuClick, admin = false, displayName, onLogout }: HeaderProps) {
  return (
    <header className="topbar">
      <button className="icon-button mobile-menu" aria-label="Open navigation" onClick={onMenuClick}><Menu size={19} /></button>
      <div className="topbar-title">
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
      </div>
      <div className="topbar-actions">
        <span className="system-pill"><span className="route-dot" /> API route</span>
        <button className="icon-button" aria-label="Notifications"><Bell size={18} /></button>
        <div className="profile-chip"><div className="avatar">{admin ? 'AD' : 'AL'}</div><span>{displayName ?? (admin ? 'Demo administrator' : 'Demo athlete')}</span><ChevronDown size={14} /></div>
        <button className="logout-button" onClick={onLogout}><LogOut size={15} /><span>Log out</span></button>
      </div>
    </header>
  )
}
