import Head from 'next/head'
import { useConfig } from 'nextra-theme-docs'
import { Hero } from '../components/Hero'
import { PipelineDemo } from '../components/PipelineDemo'
import { Features } from '../components/Features'

export default function Home() {
    return (
        <>
            <Head>
                <title>Trust, but Verify – Aroviq</title>
            </Head>
            <div className="home-content">
                <Hero />

                <div className="relative py-20 bg-black">
                    <div className="absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-[#030303] to-transparent pointer-events-none" />

                    <div className="text-center mb-16 relative z-10">
                        <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
                            The Hybrid Engine
                        </h2>
                        <p className="text-gray-400">Watch the firewall in action.</p>
                    </div>
                    <PipelineDemo />
                </div>

                <div className="py-20 bg-black">
                    <div className="text-center mb-16">
                        <h2 className="text-3xl md:text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-b from-white to-gray-500">
                            Battle Tested Features
                        </h2>
                    </div>
                    <Features />
                </div>
            </div>
        </>
    )
}
