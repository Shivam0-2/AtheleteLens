import { Activity, ArrowUpRight, ClipboardList, Dumbbell, Gauge, LayoutDashboard, Settings2, ShieldCheck } from 'lucide-react'
import { NavLink } from 'react-router-dom'

interface SidebarProps { admin?: boolean; open?: boolean; onNavigate?: () => void }

export function Sidebar({ admin = false, open = false, onNavigate }: SidebarProps) {
  const links = admin
    ? [
        { to: '/admin/diagnostics', label: 'Diagnostics', icon: Gauge },
        { to: '/login', label: 'Athlete sign-in', icon: Activity },
      ]
    : [
        { to: '/dashboard', label: 'Overview', icon: LayoutDashboard },
        { to: '/exercises', label: 'Assessments', icon: Dumbbell },
        { to: '/results', label: 'Latest results', icon: ClipboardList },
      ]

  return (
    <aside className={`sidebar ${open ? 'sidebar-open' : ''}`}>
      <NavLink to="/dashboard" className="brand" onClick={onNavigate}>
        <div className="brand-mark"><Activity size={20} strokeWidth={2.5} /></div>
        <div><strong>Athlete<span>Lens</span></strong><small>PERFORMANCE LAB</small></div>
      </NavLink>
      <div className="sidebar-label">{admin ? 'TECHNICAL' : 'WORKSPACE'}</div>
      <nav className="side-nav">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink key={to} to={to} onClick={onNavigate} className={({ isActive }) => `nav-link ${isActive ? 'nav-active' : ''}`}>
            <Icon size={18} /><span>{label}</span><ArrowUpRight className="nav-arrow" size={14} />
          </NavLink>
        ))}
      </nav>
      {!admin && <>
        <div className="sidebar-label sidebar-label-spaced">ASSESSMENT BATTERY</div>
        <div className="side-exercise-list">
          <span><i className="exercise-dot blue" />Push-up</span>
          <span><i className="exercise-dot violet" />Curl-up</span>
          <span><i className="exercise-dot green" />Vertical jump</span>
          <span><i className="exercise-dot orange" />Broad jump</span>
        </div>
      </>}
      <div className="sidebar-spacer" />
      <div className="side-notice"><ShieldCheck size={17} /><div><b>{admin ? 'Unprotected demo' : 'Research prototype'}</b><small>{admin ? 'Admin authentication is not connected.' : 'Preliminary measurements only.'}</small></div></div>
      <NavLink className="nav-link settings-link" to={admin ? '/login' : '/admin/login'}><Settings2 size={18} /><span>{admin ? 'Athlete access' : 'Admin access'}</span></NavLink>
      <div className="sidebar-footer">ATHLETELENS · ACADEMIC DEMO</div>
    </aside>
  )
}
