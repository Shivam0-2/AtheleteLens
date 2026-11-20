import { createContext, useContext, useMemo, useState, type ReactNode } from 'react'
import { authenticateDemo } from './demoCredentials'
import { clearDemoSession, readDemoSession, writeDemoSession, type DemoRole, type DemoSession } from './sessionStore'

interface AuthContextValue {
  session: DemoSession | null
  login: (role: DemoRole, email: string, password: string) => { ok: boolean; message?: string }
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function DemoAuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<DemoSession | null>(() => readDemoSession())

  const value = useMemo<AuthContextValue>(() => ({
    session,
    login(role, email, password) {
      const authenticated = authenticateDemo(role, email, password)
      if (!authenticated) return { ok: false, message: 'Those credentials do not match this demo role.' }
      try {
        writeDemoSession(authenticated)
        setSession(authenticated)
        return { ok: true }
      } catch {
        return { ok: false, message: 'This browser could not save the demo session. Check storage settings and try again.' }
      }
    },
    logout() {
      try { clearDemoSession() } catch { /* Still end the in-memory demo session. */ }
      setSession(null)
    },
  }), [session])

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useDemoAuth(): AuthContextValue {
  const context = useContext(AuthContext)
  if (!context) throw new Error('useDemoAuth must be used inside DemoAuthProvider')
  return context
}
