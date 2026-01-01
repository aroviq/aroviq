import React, { useEffect, useState, useRef } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';

interface SlotWordProps {
    finalWord: string;
    alternatives: string[];
    delay?: number;
    scrollSpeed?: number;
    className?: string;
}

const SlotWord = ({ finalWord, alternatives, delay = 0, scrollSpeed = 100, className = '' }: SlotWordProps) => {
    const [currentWord, setCurrentWord] = useState('');
    const [isScrolling, setIsScrolling] = useState(false);
    const [isSettled, setIsSettled] = useState(false);
    const ref = useRef<HTMLSpanElement>(null);
    const isInView = useInView(ref, { once: true, margin: '-100px' });
    const hasAnimated = useRef(false);

    useEffect(() => {
        if (!isInView || hasAnimated.current) return;
        hasAnimated.current = true;

        const startTimeout = setTimeout(() => {
            setIsScrolling(true);
            const allWords = [...alternatives, finalWord];
            let index = 0;

            const interval = setInterval(() => {
                if (index >= allWords.length) {
                    clearInterval(interval);
                    setCurrentWord(finalWord);
                    setIsScrolling(false);
                    setIsSettled(true);
                    return;
                }
                setCurrentWord(allWords[index]);
                index++;
            }, scrollSpeed);

            return () => clearInterval(interval);
        }, delay);

        return () => clearTimeout(startTimeout);
    }, [isInView, alternatives, finalWord, delay, scrollSpeed]);

    const displayWord = currentWord || finalWord;
    const isVisible = currentWord !== '';

    return (
        <span ref={ref} className={`inline-block relative ${className}`}>
            <AnimatePresence mode="wait">
                <motion.span
                    key={currentWord || 'empty'}
                    initial={{ y: 25, opacity: 0, filter: 'blur(6px)' }}
                    animate={{ 
                        y: 0, 
                        opacity: isVisible ? 1 : 0, 
                        filter: 'blur(0px)' 
                    }}
                    exit={{ y: -25, opacity: 0, filter: 'blur(6px)' }}
                    transition={{ duration: 0.1, ease: 'linear' }}
                    className="inline-block"
                    style={{ 
                        color: isSettled ? 'white' : '#9ca3af',
                    }}
                >
                    {displayWord}
                </motion.span>
            </AnimatePresence>
            <span className="invisible absolute">{finalWord}</span>
        </span>
    );
};

interface SlotTextProps {
    children: string;
    wordOptions: { [key: string]: string[] }; // { "Hybrid": ["Mixed", "Combined", "Fusion"], ... }
    className?: string;
    staggerDelay?: number;
    scrollSpeed?: number;
}

export const SlotText = ({ children, wordOptions, className = '', staggerDelay = 200, scrollSpeed = 100 }: SlotTextProps) => {
    const words = children.split(' ');

    return (
        <span className={className}>
            {words.map((word, index) => {
                const alternatives = wordOptions[word] || [];

                return (
                    <React.Fragment key={index}>
                        <SlotWord
                            finalWord={word}
                            alternatives={alternatives}
                            delay={index * staggerDelay}
                            scrollSpeed={scrollSpeed}
                        />
                        {index < words.length - 1 && ' '}
                    </React.Fragment>
                );
            })}
        </span>
    );
};

// Keep the old DecryptText for backwards compatibility
const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%&*<>[]{}';

interface DecryptTextProps {
    text: string;
    className?: string;
    delay?: number;
    speed?: number;
}

export const DecryptText = ({ text, className = '', delay = 0, speed = 50 }: DecryptTextProps) => {
    const [displayText, setDisplayText] = useState('');
    const ref = useRef<HTMLSpanElement>(null);
    const isInView = useInView(ref, { once: true, margin: '-100px' });
    const hasAnimated = useRef(false);

    useEffect(() => {
        if (!isInView || hasAnimated.current) return;
        hasAnimated.current = true;

        const startDecrypt = setTimeout(() => {
            let iteration = 0;
            const maxIterations = text.length;

            setDisplayText(
                text.split('').map(char => char === ' ' ? ' ' : CHARS[Math.floor(Math.random() * CHARS.length)]).join('')
            );

            const interval = setInterval(() => {
                setDisplayText(
                    text
                        .split('')
                        .map((char, index) => {
                            if (char === ' ') return ' ';
                            if (index < iteration) return char;
                            return CHARS[Math.floor(Math.random() * CHARS.length)];
                        })
                        .join('')
                );

                iteration += 1 / 3;

                if (iteration >= maxIterations) {
                    clearInterval(interval);
                    setDisplayText(text);
                }
            }, speed);

            return () => clearInterval(interval);
        }, delay);

        return () => clearTimeout(startDecrypt);
    }, [isInView, text, delay, speed]);

    return (
        <motion.span
            ref={ref}
            className={className}
            initial={{ opacity: 0 }}
            animate={{ opacity: isInView ? 1 : 0 }}
            transition={{ duration: 0.3 }}
        >
            {displayText || text}
        </motion.span>
    );
};
