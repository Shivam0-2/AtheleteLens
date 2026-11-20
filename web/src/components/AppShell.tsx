import { useState } from 'react'
import { Outlet, useLocation, useNavigate } from 'react-router-dom'
import { ShieldCheck } from 'lucide-react'
import { useDemoAuth } from '../auth/AuthContext'
import { Header } from './Header'
import { Sidebar } from './Sidebar'

const titles: Record<string, { title: string; eyebrow?: string }> = {
  '/dashboard': { title: 'Your performance overview' },
  '/exercises': { title: 'Assessment library' },
  '/results': { title: 'Assessment results' },
}

export function AppShell({ admin = false }: { admin?: boolean }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const location = useLocation()
  const navigate = useNavigate()
  const { session, logout } = useDemoAuth()
  const title = admin ? 'System diagnostics' : location.pathname.startsWith('/results') ? 'Assessment results' : titles[location.pathname]?.title ?? 'Assessment setup'

  function signOut() {
    logout()
    navigate(admin ? '/admin/login' : '/login', { replace: true })
  }

  return (
    <div className={`app-layout ${admin ? 'admin-layout' : ''}`}>
      <Sidebar admin={admin} open={menuOpen} onNavigate={() => setMenuOpen(false)} />
      {menuOpen && <button className="sidebar-scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}
      <main className="main-area">
        <Header title={title} eyebrow={admin ? 'ADMIN CONSOLE · DEMO ONLY' : undefined} admin={admin} displayName={session?.displayName} onLogout={signOut} onMenuClick={() => setMenuOpen((value) => !value)} />
        {admin && <div className="admin-banner"><ShieldCheck size={16} /> Demo role checks run in this browser only. Diagnostics shows local submissions, not server-side admin data.</div>}
        <div className="page-content"><Outlet /></div>
      </main>
    </div>
  )
}
