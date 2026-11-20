import { useState, type FormEvent } from 'react'
import { Activity, ArrowLeft, LockKeyhole } from 'lucide-react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useDemoAuth } from '../auth/AuthContext'
import { DEMO_CREDENTIALS } from '../auth/demoCredentials'
import type { DemoRole } from '../auth/sessionStore'

export function AuthPage({ mode = 'login', admin = false }: { mode?: 'login' | 'register'; admin?: boolean }) {
  const role: DemoRole = admin ? 'admin' : 'athlete'
  const register = mode === 'register'
  const credentials = DEMO_CREDENTIALS[role]
  const { session, login } = useDemoAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)

  function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const result = login(role, email, password)
    if (!result.ok) {
      setError(result.message ?? 'Demo sign-in failed.')
      return
    }
    const requested = (location.state as { from?: string } | null)?.from
    const safeDestination = requested?.startsWith(role === 'admin' ? '/admin' : '/dashboard')
      ? requested
      : role === 'admin' ? '/admin/diagnostics' : '/dashboard'
    navigate(safeDestination, { replace: true })
  }

  const title = admin ? 'Administrator sign-in' : register ? 'Create your account' : 'Athlete sign-in'
  return <div className="auth-page"><div className="auth-decoration"><div className="auth-orbit orbit-a" /><div className="auth-orbit orbit-b" /><div className="auth-mark"><Activity size={38} /></div><div className="auth-quote"><span className="eyebrow">ATHLETELENS PERFORMANCE LAB</span><h2>See the movement<br /><em>behind the moment.</em></h2><p>A preliminary computer-vision prototype for exploring sports movement.</p></div><span className="auth-footnote">ACADEMIC DEMONSTRATION · DEMO ACCESS</span></div><main className="auth-main"><Link to="/" className="auth-brand"><span className="brand-mark"><Activity size={18} /></span><b>AthleteLens</b></Link><section className="auth-card"><Link to={admin ? '/admin/login' : '/login'} className="back-link"><ArrowLeft size={15} /> {admin ? 'Administrator access' : 'Athlete workspace'}</Link><div className="auth-title"><span className="eyebrow">{admin ? 'ADMINISTRATOR WORKSPACE' : 'ATHLETE WORKSPACE'}</span><h1>{title}</h1><p>{register ? 'Account registration is not connected. Use the demo athlete login instead.' : 'This role-based access is a browser-only academic demo, not production authentication.'}</p></div>
      {register ? <div className="auth-placeholder"><div className="auth-placeholder-icon"><LockKeyhole size={19} /></div><div><strong>Registration is unavailable</strong><p>No user accounts are created or stored by this prototype.</p></div></div> : <>
        <div className="demo-credential-card"><div className="credential-title"><LockKeyhole size={15} /><strong>Demo {role} credentials</strong><span>PUBLIC DEMO</span></div><div><small>Email</small><code>{credentials.email}</code></div><div><small>Password</small><code>{credentials.password}</code></div><button type="button" className="text-button" onClick={() => { setEmail(credentials.email); setPassword(credentials.password); setError(null) }}>Fill demo credentials</button></div>
        <form className="auth-form" onSubmit={handleLogin}>
          <label>Email<input type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
          <label>Password<input type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
          {error && <div className="form-error" role="alert">{error}</div>}
          <button className="button button-primary auth-submit" type="submit">Continue as {role === 'admin' ? 'administrator' : 'athlete'}</button>
        </form>
      </>}
      {session && session.role !== role && !register && <p className="role-switch-note">Signing in here will switch this browser from the {session.role} demo session to the {role} demo session.</p>}
      <div className="auth-switch">{admin ? <>Athlete access? <Link to="/login">Go to athlete login</Link></> : register ? <>Already have demo access? <Link to="/login">Athlete login</Link></> : <>Administrator? <Link to="/admin/login">Go to administrator login</Link></>}</div>
      <div className="auth-demo-note">Demo role checks run in this browser and are not secure. Do not use real credentials.</div>
    </section><div className="auth-bottom">Sessions are stored locally in this browser. No backend identity service is connected.</div></main></div>
}
