import type { ReactNode } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { DemoAuthProvider, useDemoAuth } from './auth/AuthContext'
import type { DemoRole } from './auth/sessionStore'
import { AppShell } from './components/AppShell'
import { AdminDiagnosticsPage } from './pages/AdminDiagnosticsPage'
import { AssessmentPage } from './pages/AssessmentPage'
import { AuthPage } from './pages/AuthPages'
import { DashboardPage } from './pages/DashboardPage'
import { ExercisesPage } from './pages/ExercisesPage'
import { ResultsPage } from './pages/ResultsPage'

function RoleLanding() {
  const { session } = useDemoAuth()
  return <Navigate to={!session ? '/login' : session.role === 'admin' ? '/admin/diagnostics' : '/dashboard'} replace />
}

function RequireRole({ role, children }: { role: DemoRole; children: ReactNode }) {
  const { session } = useDemoAuth()
  const location = useLocation()
  if (!session) return <Navigate to={role === 'admin' ? '/admin/login' : '/login'} state={{ from: location.pathname }} replace />
  if (session.role !== role) return <Navigate to={role === 'admin' ? '/admin/login' : '/login'} state={{ from: location.pathname }} replace />
  return children
}

function AppRoutes() {
  return <Routes>
    <Route path="/" element={<RoleLanding />} />
    <Route path="/login" element={<AuthPage />} />
    <Route path="/register" element={<AuthPage mode="register" />} />
    <Route element={<RequireRole role="athlete"><AppShell /></RequireRole>}>
      <Route path="/dashboard" element={<DashboardPage />} />
      <Route path="/exercises" element={<ExercisesPage />} />
      <Route path="/assessments/:exercise" element={<AssessmentPage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/results/:recordId" element={<ResultsPage />} />
    </Route>
    <Route path="/admin/login" element={<AuthPage admin />} />
    <Route element={<RequireRole role="admin"><AppShell admin /></RequireRole>}>
      <Route path="/admin/diagnostics" element={<AdminDiagnosticsPage />} />
    </Route>
    <Route path="*" element={<RoleLanding />} />
  </Routes>
}

export default function App() {
  return <DemoAuthProvider><AppRoutes /></DemoAuthProvider>
}
