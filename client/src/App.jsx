import { ToastProvider } from './components/ui/toast'
import { Toaster } from './components/ui/toaster'
import Dashboard from './pages/Dashboard'

function App() {
  return (
    <ToastProvider>
      <div className="min-h-screen bg-background">
        <Dashboard />
        <Toaster />
      </div>
    </ToastProvider>
  )
}

export default App 