import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, ShieldAlert, ShieldCheck, Zap, Brain, Cpu, Activity } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

type Step = 'AGENT' | 'TIER0' | 'TIER1' | 'DECISION';
type Verdict = 'PASS' | 'BLOCK';

export const PipelineDemo = () => {
    const [activeStep, setActiveStep] = useState<Step>('AGENT');
    const [verdict, setVerdict] = useState<Verdict>('PASS');
    const [packetId, setPacketId] = useState(0);

    // Simulation loop (same logic, visual upgrade)
    useEffect(() => {
        let timeout: NodeJS.Timeout;
        const runSimulation = async () => {
            setActiveStep('AGENT');
            const isBadActor = Math.random() > 0.7;
            await new Promise(r => timeout = setTimeout(r, 1000));
            setActiveStep('TIER0');
            await new Promise(r => timeout = setTimeout(r, 600));

            if (isBadActor && Math.random() > 0.5) {
                setVerdict('BLOCK');
                setActiveStep('DECISION');
                await new Promise(r => timeout = setTimeout(r, 2000));
                setVerdict('PASS'); setPacketId(p => p + 1); return;
            }
            setActiveStep('TIER1');
            await new Promise(r => timeout = setTimeout(r, 1500));
            setVerdict(isBadActor ? 'BLOCK' : 'PASS');
            setActiveStep('DECISION');
            await new Promise(r => timeout = setTimeout(r, 2000));
            setVerdict('PASS'); setPacketId(p => p + 1);
        };
        runSimulation();
        return () => clearTimeout(timeout);
    }, [packetId]);

    return (
        <div className="w-full max-w-6xl mx-auto px-4 py-20 relative">
            <div className={clsx(
                "relative p-12 rounded-3xl border border-white/10 overflow-hidden bg-black/40 backdrop-blur-xl transition-colors duration-500",
                verdict === 'BLOCK' && activeStep === 'DECISION' ? "shadow-[0_0_100px_rgba(239,68,68,0.2)] border-red-900/50" : "shadow-2xl"
            )}>
                {/* Background Grid */}
                <div className="absolute inset-0 bg-grid-white opacity-[0.03]" />

                <div className="flex flex-col lg:flex-row justify-between items-center gap-12 relative z-10">
                    {/* Connection Line */}
                    <div className="absolute top-1/2 left-20 right-20 h-[2px] bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 -z-10 hidden lg:block" />

                    <GlassNode
                        icon={<Cpu className="w-6 h-6" />}
                        label="Agent Output"
                        active={activeStep === 'AGENT'}
                        color="text-white"
                    />

                    <GlassNode
                        icon={<Zap className="w-6 h-6" />}
                        label="Tier 0"
                        sub="Regex/Symbolic"
                        active={activeStep === 'TIER0'}
                        color="text-aroviq-cyan"
                        glowColor="rgba(0, 242, 234, 0.5)"
                    />

                    <GlassNode
                        icon={<Brain className="w-6 h-6" />}
                        label="Tier 1"
                        sub="LLM Judge"
                        active={activeStep === 'TIER1'}
                        color="text-aroviq-purple"
                        glowColor="rgba(112, 0, 255, 0.5)"
                    />

                    <div className="relative group">
                        <motion.div
                            animate={{
                                scale: activeStep === 'DECISION' ? 1.1 : 1,
                                background: activeStep === 'DECISION'
                                    ? (verdict === 'PASS' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)')
                                    : 'rgba(255,255,255,0.05)',
                                boxShadow: activeStep === 'DECISION'
                                    ? (verdict === 'PASS' ? '0 0 40px rgba(16, 185, 129, 0.5)' : '0 0 40px rgba(239, 68, 68, 0.5)')
                                    : 'none'
                            }}
                            className="w-24 h-24 rounded-2xl border border-white/10 backdrop-blur-xl flex items-center justify-center transition-all duration-300"
                        >
                            {activeStep === 'DECISION' ? (
                                verdict === 'PASS' ? <ShieldCheck className="w-10 h-10 text-emerald-500" />
                                    : <ShieldAlert className="w-10 h-10 text-red-500" />
                            ) : <Shield className="w-10 h-10 text-gray-500" />}
                        </motion.div>
                        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-4 text-center w-max">
                            <div className={twMerge("font-bold text-sm tracking-wide transition-colors", activeStep === 'DECISION' ? "text-white" : "text-gray-600")}>
                                Decision
                            </div>
                            <div className="text-xs text-gray-500 font-mono mt-1">
                                {activeStep === 'DECISION' ? (verdict === 'PASS' ? 'Approved' : 'Blocked') : 'Pending'}
                            </div>
                        </div>
                    </div>

                    {/* Glowing Particle */}
                    <motion.div
                        className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full z-20 hidden lg:block"
                        style={{
                            background: 'radial-gradient(circle, #fff 0%, transparent 70%)',
                            boxShadow: '0 0 20px 5px rgba(255,255,255,0.5)'
                        }}
                        initial={{ left: '5%', opacity: 1 }}
                        animate={{
                            left: activeStep === 'AGENT' ? '5%' :
                                activeStep === 'TIER0' ? '35%' :
                                    activeStep === 'TIER1' ? '65%' : '90%',
                            opacity: activeStep === 'DECISION' ? 0 : 1
                        }}
                        transition={{ duration: 0.6, ease: "circInOut" }}
                    />
                </div>

                <div className="flex justify-center mt-20 gap-8 text-center text-sm font-mono text-gray-400">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-aroviq-cyan animate-pulse" />
                        Tier 0: &lt;1ms
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-aroviq-purple animate-pulse" />
                        Tier 1: ~200ms
                    </div>
                </div>
            </div>
        </div>
    )
}

const GlassNode = ({ icon, label, sub, active, color, glowColor = 'rgba(255,255,255,0.2)' }: any) => {
    return (
        <div className="relative group">
            <motion.div
                animate={{
                    borderColor: active ? color.replace('text-', 'border-') : 'rgba(255,255,255,0.05)',
                    boxShadow: active ? `0 0 40px ${glowColor}` : 'none'
                }}
                className={twMerge(
                    "w-24 h-24 rounded-2xl bg-white/5 backdrop-blur-md border border-white/10 flex items-center justify-center transition-all duration-500",
                    active ? color : "text-gray-600"
                )}
            >
                {active && (
                    <motion.div
                        className="absolute inset-0 bg-white/5 rounded-2xl"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0, 0.5, 0] }}
                        transition={{ repeat: Infinity, duration: 1.5 }}
                    />
                )}
                {icon}
            </motion.div>
            <div className="absolute top-full left-1/2 -translate-x-1/2 mt-4 text-center w-max">
                <div className={twMerge("font-bold text-sm tracking-wide transition-colors", active ? "text-white" : "text-gray-600")}>
                    {label}
                </div>
                {sub && <div className="text-xs text-gray-500 font-mono mt-1">{sub}</div>}
            </div>
        </div>
    )
}
