import { afterEach, describe, expect, it, vi } from 'vitest'
import { authenticateDemo } from './demoCredentials'
import { clearDemoSession, readDemoSession, writeDemoSession } from './sessionStore'

function memoryStorage() {
  const values = new Map<string, string>()
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { values.set(key, value) },
    removeItem: (key: string) => { values.delete(key) },
  }
}

afterEach(() => vi.unstubAllGlobals())

describe('demo session', () => {
  it('authenticates only credentials for the selected demo role', () => {
    expect(authenticateDemo('athlete', 'athlete@demo.athletelens', 'athlete-demo')?.role).toBe('athlete')
    expect(authenticateDemo('admin', 'admin@demo.athletelens', 'admin-demo')?.role).toBe('admin')
    expect(authenticateDemo('admin', 'athlete@demo.athletelens', 'athlete-demo')).toBeNull()
    expect(authenticateDemo('athlete', 'athlete@demo.athletelens', 'wrong')).toBeNull()
  })

  it('persists the role session across reads and removes it on logout', () => {
    vi.stubGlobal('localStorage', memoryStorage())
    const session = authenticateDemo('athlete', 'athlete@demo.athletelens', 'athlete-demo')!
    writeDemoSession(session)
    expect(readDemoSession()).toEqual(session)
    clearDemoSession()
    expect(readDemoSession()).toBeNull()
  })

  it('ignores malformed stored sessions', () => {
    const storage = memoryStorage()
    storage.setItem('athletelens.demo-session.v1', '{bad json')
    vi.stubGlobal('localStorage', storage)
    expect(readDemoSession()).toBeNull()
  })
})
