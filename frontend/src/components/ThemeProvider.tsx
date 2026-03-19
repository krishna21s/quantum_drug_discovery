import { createContext, useContext, useEffect, useState, ReactNode } from "react";

type Theme = "dark" | "light";

interface ThemeContextValue {
    theme: Theme;
    toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({
    theme: "dark",
    toggleTheme: () => { },
});

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [theme, setTheme] = useState<Theme>(() => {
        if (typeof window !== "undefined") {
            const stored = localStorage.getItem("qpharmx-theme") as Theme | null;
            if (stored) return stored;
        }
        return "dark";
    });

    useEffect(() => {
        const root = document.documentElement;
        root.classList.remove("dark", "light");
        root.classList.add(theme);
        localStorage.setItem("qpharmx-theme", theme);
    }, [theme]);

    const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

    return (
        <ThemeContext.Provider value={{ theme, toggleTheme }}>
            {children}
        </ThemeContext.Provider>
    );
}

export function useTheme() {
    return useContext(ThemeContext);
}
