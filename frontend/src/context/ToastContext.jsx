import React, { createContext, useContext, useState, useCallback } from 'react'

const ToastContext = createContext(null)

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const toast = useCallback((msg, type = 'info', duration = 3500) => {
    const id = Date.now() + Math.random()
    setToasts(t => [...t, { id, msg, type }])
    setTimeout(() => setToasts(t => t.filter(x => x.id !== id)), duration)
  }, [])

  const dismiss = (id) => setToasts(t => t.filter(x => x.id !== id))

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none">
        {toasts.map(t => {
          const styles = {
            success: { bg: 'rgba(34,197,94,0.12)',  border: 'rgba(34,197,94,0.35)',  text: '#86efac',  icon: '✓' },
            error:   { bg: 'rgba(239,68,68,0.12)',  border: 'rgba(239,68,68,0.35)',  text: '#fca5a5',  icon: '✗' },
            warn:    { bg: 'rgba(234,179,8,0.12)',  border: 'rgba(234,179,8,0.35)',  text: '#fde047',  icon: '⚠' },
            info:    { bg: 'rgba(200,169,110,0.12)', border: 'rgba(200,169,110,0.35)', text: '#c8a96e', icon: 'ℹ' },
          }
          const s = styles[t.type] || styles.info
          return (
            <div key={t.id}
              className="pointer-events-auto flex items-center gap-3 px-4 py-3 rounded-xl text-sm animate-fadeUp"
              style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.text, backdropFilter: 'blur(12px)', minWidth: 260, maxWidth: 380 }}
              onClick={() => dismiss(t.id)}>
              <span className="flex-shrink-0 font-bold">{s.icon}</span>
              <span className="flex-1">{t.msg}</span>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  return useContext(ToastContext)
}
