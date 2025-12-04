// tailwind.config.js
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        gmail: {
          red: '#D9302C',        // primary accent (Gmail red)
          redDark: '#B1271F',
          surface: '#FFFFFF',
          bg: '#F1F3F4',         // gmail-like background
          muted: '#5F6368',
          lightGray: '#F5F7F9',
        },
      },
      boxShadow: {
        'gmail-card': '0 1px 2px rgba(60,64,67,0.12), 0 1px 3px rgba(60,64,67,0.08)'
      },
      borderRadius: {
        'gmail-sm': '10px'
      }
    },
  },
  plugins: [],
}
