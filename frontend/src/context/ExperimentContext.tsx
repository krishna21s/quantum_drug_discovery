/**
 * Experiment Context — Session Data Management
 * ================================================
 * Stores experiment results in React Context + sessionStorage.
 * All pages read from this context for consistent data flow.
 * Data survives page refresh via sessionStorage, clears on tab close.
 */

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import type { GenerateResponse, GenerateRequest } from "@/lib/drugApi";

// ── Session Experiment Shape ──
export interface ExperimentSession {
  config: GenerateRequest;
  result: GenerateResponse;
  timestamp: string;  // ISO string
}

// ── Context Interface ──
interface ExperimentContextType {
  session: ExperimentSession | null;
  saveSession: (config: GenerateRequest, result: GenerateResponse) => void;
  clearSession: () => void;
  hasSession: boolean;
}

const ExperimentContext = createContext<ExperimentContextType>({
  session: null,
  saveSession: () => {},
  clearSession: () => {},
  hasSession: false,
});

const SESSION_KEY = "qpharmx_experiment_session";

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<ExperimentSession | null>(() => {
    try {
      const stored = sessionStorage.getItem(SESSION_KEY);
      return stored ? JSON.parse(stored) : null;
    } catch {
      return null;
    }
  });

  // Sync to sessionStorage on every change
  useEffect(() => {
    if (session) {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
    } else {
      sessionStorage.removeItem(SESSION_KEY);
    }
  }, [session]);

  const saveSession = useCallback((config: GenerateRequest, result: GenerateResponse) => {
    const newSession: ExperimentSession = {
      config,
      result,
      timestamp: new Date().toISOString(),
    };
    setSession(newSession);
  }, []);

  const clearSession = useCallback(() => {
    setSession(null);
  }, []);

  return (
    <ExperimentContext.Provider value={{
      session,
      saveSession,
      clearSession,
      hasSession: session !== null,
    }}>
      {children}
    </ExperimentContext.Provider>
  );
}

export function useExperiment() {
  return useContext(ExperimentContext);
}
