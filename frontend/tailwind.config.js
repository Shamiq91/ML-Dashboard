/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', 'ui-monospace', 'monospace'],
        display: ['"Chakra Petch"', '"JetBrains Mono"', 'sans-serif'],
      },
      colors: {
        carbon: { 950: '#07080a', 900: '#0c0e11', 800: '#12151a', 700: '#1a1e25', 600: '#252a33', 500: '#39404b', 400: '#667080', 300: '#97a1b0', 200: '#c3c9d3', 100: '#e8ebf0' },
        sector: { purple: '#b45bff', green: '#34e27a', yellow: '#ffd23f', red: '#ff4d57', cyan: '#3ec6ff', orange: '#ff9a3c' },
      },
      keyframes: {
        rise: { from: { opacity: 0, transform: 'translateY(6px)' }, to: { opacity: 1, transform: 'none' } },
        pulseDot: { '0%,100%': { opacity: 1 }, '50%': { opacity: 0.35 } },
        sweep: { from: { backgroundPosition: '200% 0' }, to: { backgroundPosition: '-200% 0' } },
      },
      animation: { rise: 'rise .35s ease-out both', pulseDot: 'pulseDot 1.2s ease-in-out infinite', sweep: 'sweep 2.2s linear infinite' },
    },
  },
  plugins: [],
}
