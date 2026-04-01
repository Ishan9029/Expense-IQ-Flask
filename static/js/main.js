const storageKey = "expenseiq-theme";

function applyTheme(theme) {
    const resolvedTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", resolvedTheme);

    const toggle = document.getElementById("theme-toggle");
    if (!toggle) return;

    const icon = toggle.querySelector(".theme-toggle-icon");
    const text = toggle.querySelector(".theme-toggle-text");

    if (resolvedTheme === "dark") {
        if (icon) icon.textContent = "☀️";
        if (text) text.textContent = "Light";
        toggle.setAttribute("aria-label", "Switch to light mode");
        toggle.setAttribute("title", "Switch to light mode");
    } else {
        if (icon) icon.textContent = "🌙";
        if (text) text.textContent = "Dark";
        toggle.setAttribute("aria-label", "Switch to dark mode");
        toggle.setAttribute("title", "Switch to dark mode");
    }
}

function getPreferredTheme() {
    const savedTheme = localStorage.getItem(storageKey);
    if (savedTheme === "light" || savedTheme === "dark") {
        return savedTheme;
    }

    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

document.addEventListener("DOMContentLoaded", () => {
    applyTheme(getPreferredTheme());

    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
        toggle.addEventListener("click", () => {
            const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
            const nextTheme = currentTheme === "dark" ? "light" : "dark";
            localStorage.setItem(storageKey, nextTheme);
            applyTheme(nextTheme);
        });
    }

    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    mediaQuery.addEventListener?.("change", (event) => {
        if (!localStorage.getItem(storageKey)) {
            applyTheme(event.matches ? "dark" : "light");
        }
    });
});
