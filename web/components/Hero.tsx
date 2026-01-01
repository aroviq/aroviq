import React, { useRef, useState, useEffect } from 'react';
import Link from 'next/link';
import { motion, useMotionTemplate, useMotionValue, animate } from 'framer-motion';
import { ArrowRight, BarChart2, Shield } from 'lucide-react';

const COLORS = ["#00f2ea", "#7000ff", "#00f2ea"];

export const Hero = () => {
    const color = useMotionValue(COLORS[0]);
    const backgroundImage = useMotionTemplate`radial-gradient(125% 125% at 50% 10%, #030303 50%, ${color})`;
    const border = useMotionTemplate`1px solid ${color}`;
    const boxShadow = useMotionTemplate`0px 4px 24px ${color}`;

    useEffect(() => {
        animate(color, COLORS, {
            ease: "easeInOut",
            duration: 5,
            repeat: Infinity,
            repeatType: "mirror",
        });
    }, [color]);

    return (
        <motion.section
            style={{ backgroundImage }}
            className="relative flex flex-col items-center justify-center min-h-screen px-4 py-24 text-center overflow-hidden"
        >
            <div className="absolute inset-0 bg-grid-white opacity-[0.05]" />

            {/* Logo Section - Bigger & Centered, No Glow */}
            <div className="relative mb-12 flex justify-center items-center w-full">
                <motion.img
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 1 }}
                    src="/logo.png"
                    alt="Aroviq Logo"
                    className="relative z-10 w-64 h-auto md:w-80 lg:w-96 max-w-full object-contain"
                />
            </div>

            <div className="z-10 max-w-5xl mx-auto space-y-8 flex flex-col items-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.2 }}
                    className="flex justify-center"
                >
                    <span className="px-3 py-1 text-xs font-mono font-medium text-aroviq-cyan bg-aroviq-cyan/10 border border-aroviq-cyan/20 rounded-full backdrop-blur-md">
                        v0.3.0 Now Available
                    </span>
                </motion.div>

                <motion.h1
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.4 }}
                    className="text-4xl md:text-6xl lg:text-7xl font-black tracking-tighter text-center"
                >
                    <span className="text-white">Trust, but </span>
                    <span className="bg-clip-text text-transparent bg-gradient-to-r from-aroviq-cyan via-white to-aroviq-purple text-glow">
                        Verify.
                    </span>
                </motion.h1>

                <motion.p
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.6 }}
                    className="text-xl md:text-2xl text-gray-400 max-w-2xl mx-auto leading-relaxed text-center"
                >
                    The <span className="text-white font-semibold">&lt;1ms</span> firewall for AI Agents.
                    Deterministic safety meets state-of-the-art probabilistic reasoning.
                </motion.p>

                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8, delay: 0.8 }}
                    className="flex flex-col sm:flex-row gap-6 justify-center items-center pt-8 w-full"
                >
                    <motion.div
                        style={{ border, boxShadow }}
                        className="rounded-full overflow-hidden"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                    >
                        <Link href="/docs" className="flex items-center gap-2 px-8 py-4 bg-black text-white font-bold relative z-10">
                            Get Started <ArrowRight className="w-4 h-4" />
                        </Link>
                    </motion.div>

                    <Link href="/benchmarks" className="px-8 py-4 text-gray-400 hover:text-white transition-colors flex items-center gap-2 font-medium">
                        <BarChart2 className="w-4 h-4" /> View Benchmarks
                    </Link>
                </motion.div>
            </div>

            {/* Bottom Fade */}
            <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#030303] to-transparent pointer-events-none" />
        </motion.section>
    );
};
