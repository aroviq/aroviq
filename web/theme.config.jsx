import React from 'react'

const config = {
    logo: (
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <img src="/logonotext.png" alt="Aroviq" width={32} height={32} />
            <span style={{ fontWeight: 800, fontSize: '1.2rem', letterSpacing: '-0.02em' }}>Aroviq</span>
        </div>
    ),
    project: {
        link: 'https://github.com/ShyamSathish005/aroviq',
    },
    docsRepositoryBase: 'https://github.com/ShyamSathish005/aroviq/tree/main/web',
    useNextSeoProps() {
        return {
            titleTemplate: '%s – Aroviq'
        }
    },
    primaryHue: {
        dark: 180,
        light: 270
    },
    footer: {
        text: (
            <span>
                MIT {new Date().getFullYear()} © <a href="https://aroviq.dev" target="_blank">Aroviq</a>.
                <br />
                <span style={{ opacity: 0.5, fontSize: '0.8rem' }}>Trust, but Verify.</span>
            </span>
        )
    },
    darkMode: true,
    nextThemes: {
        defaultTheme: 'system',
        // Force dark mode for that premium feel
    },
    navbar: {
        extraContent: (
            <a href="https://github.com/ShyamSathish005/aroviq" target="_blank" className="p-2 bg-white/10 rounded-lg hover:bg-white/20 transition-colors">
                <span className="sr-only">GitHub</span>
            </a>
        )
    }
}

export default config
