export type DemoRole = 'athlete' | 'admin'

export interface DemoSession {
  role: DemoRole
  email: string
  displayName: string
  signedInAt: string
}

export const SESSION_STORAGE_KEY = 'athletelens.demo-session.v1'

function isDemoRole(value: unknown): value is DemoRole {
  return value === 'athlete' || value === 'admin'
}

export function readDemoSession(): DemoSession | null {
  try {
    const stored = localStorage.getItem(SESSION_STORAGE_KEY)
    if (!stored) return null
    const value: unknown = JSON.parse(stored)
    if (typeof value !== 'object' || value === null) return null
    const session = value as Record<string, unknown>
    if (!isDemoRole(session.role) || typeof session.email !== 'string'
        || typeof session.displayName !== 'string' || typeof session.signedInAt !== 'string') return null
    return session as unknown as DemoSession
  } catch {
    return null
  }
}

export function writeDemoSession(session: DemoSession): void {
  localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(session))
}

export function clearDemoSession(): void {
  localStorage.removeItem(SESSION_STORAGE_KEY)
}
