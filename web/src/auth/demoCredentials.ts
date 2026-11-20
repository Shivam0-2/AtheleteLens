import type { DemoRole, DemoSession } from './sessionStore'

export interface DemoCredential {
  role: DemoRole
  email: string
  password: string
  displayName: string
}

export const DEMO_CREDENTIALS: Record<DemoRole, DemoCredential> = {
  athlete: {
    role: 'athlete',
    email: 'athlete@demo.athletelens',
    password: 'athlete-demo',
    displayName: 'Demo athlete',
  },
  admin: {
    role: 'admin',
    email: 'admin@demo.athletelens',
    password: 'admin-demo',
    displayName: 'Demo administrator',
  },
}

/** Demo-only credential check. It runs entirely in the browser. */
export function authenticateDemo(
  role: DemoRole,
  email: string,
  password: string,
  now = new Date().toISOString(),
): DemoSession | null {
  const credential = DEMO_CREDENTIALS[role]
  if (email.trim().toLowerCase() !== credential.email || password !== credential.password) return null
  return { role, email: credential.email, displayName: credential.displayName, signedInAt: now }
}
