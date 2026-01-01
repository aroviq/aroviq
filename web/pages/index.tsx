import Head from 'next/head'
import { Navbar } from '../components/Navbar'
import { Hero } from '../components/Hero'
import { PipelineDemo } from '../components/PipelineDemo'
import { Features } from '../components/Features'
import { SlotText } from '../components/DecryptText'

export default function Home() {
    return (
        <>
            <Head>
                <title>Trust, but Verify – Aroviq</title>
            </Head>
            <Navbar />
            <div className="home-content pt-16">
                <Hero />

                <div className="relative py-20 bg-black">
                    <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-[#030303] to-transparent pointer-events-none" />

                    <div className="text-center mb-16 relative z-10">
                        <h2 className="text-3xl md:text-5xl font-bold mb-4">
                            <SlotText
                                wordOptions={{
                                    'The': ['A', 'An', 'One', 'Your', 'Our'],
                                    'Hybrid': ['Mixed', 'Combined', 'Fusion', 'Blended', 'Unified', 'Dual', 'Multi', 'Adaptive'],
                                    'Engine': ['System', 'Core', 'Framework', 'Pipeline', 'Module', 'Layer', 'Stack']
                                }}
                                staggerDelay={350}
                                scrollSpeed={90}
                            >
                                The Hybrid Engine
                            </SlotText>
                        </h2>
                        <p className="text-gray-400">Watch the firewall in action.</p>
                    </div>
                    <PipelineDemo />
                </div>

                <div className="py-20 bg-black">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-4">
                            <SlotText
                                wordOptions={{
                                    'Battle': ['Combat', 'Field', 'War', 'Stress', 'Load', 'Real', 'Production', 'Live'],
                                    'Tested': ['Proven', 'Verified', 'Checked', 'Validated', 'Confirmed', 'Approved', 'Certified'],
                                    'Features': ['Tools', 'Modules', 'Capabilities', 'Functions', 'Powers', 'Utilities', 'Components']
                                }}
                                staggerDelay={350}
                                scrollSpeed={90}
                            >
                                Battle Tested Features
                            </SlotText>
                        </h2>
                    </div>
                    <Features />
                </div>
            </div>
        </>
    )
}
