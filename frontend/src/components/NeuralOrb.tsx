import React, { useEffect, useRef } from "react";
import { BrainState } from "../types";

interface NeuralOrbProps {
  state: BrainState;
  size?: number;
  onClick?: () => void;
  title?: string;
}

export const NeuralOrb: React.FC<NeuralOrbProps> = ({
  state,
  size = 46,
  onClick,
  title = "Xyvora Neural Core — Click to pulse",
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const pulseRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);

    let animId = 0;
    let t = 0;

    const nodes = Array.from({ length: 14 }, (_, i) => ({
      angle: (i / 14) * Math.PI * 2,
      radiusRatio: 0.24 + (i % 3) * 0.09,
      speed: (i % 2 === 0 ? 1 : -1) * (0.012 + (i % 4) * 0.004),
    }));

    const render = () => {
      const speedMult =
        state === "thinking" ? 3.2 : state === "streaming" ? 2.3 : state === "listening" ? 2.6 : 1.0;
      t += 0.025 * speedMult;
      pulseRef.current *= 0.92;

      ctx.clearRect(0, 0, size, size);
      const cx = size / 2;
      const cy = size / 2;
      const rBase = size * 0.36 + pulseRef.current * (size * 0.08);

      const primary =
        state === "listening"
          ? "52, 211, 153" // emerald
          : state === "thinking"
          ? "251, 191, 36" // amber-gold
          : state === "streaming"
          ? "129, 140, 248" // electric indigo
          : "34, 211, 238"; // bioluminescent cyan

      const secondary = state === "thinking" ? "34, 211, 238" : "99, 102, 241";

      // Ambient radial aura
      const grad = ctx.createRadialGradient(cx, cy, size * 0.05, cx, cy, size * 0.48);
      grad.addColorStop(0, `rgba(${primary}, 0.85)`);
      grad.addColorStop(0.45, `rgba(${secondary}, 0.28)`);
      grad.addColorStop(1, "rgba(5, 8, 17, 0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, size * 0.48, 0, Math.PI * 2);
      ctx.fill();

      // Harmonic wave ring
      ctx.strokeStyle = `rgba(${primary}, 0.65)`;
      ctx.lineWidth = Math.max(1.2, size * 0.028);
      ctx.beginPath();
      for (let a = 0; a <= Math.PI * 2 + 0.1; a += 0.12) {
        const wave =
          Math.sin(a * 4 + t * 2) * (size * 0.032 * speedMult * 0.6) +
          Math.cos(a * 3 - t * 1.5) * (size * 0.02);
        const rad = rBase + wave;
        const px = cx + Math.cos(a) * rad;
        const py = cy + Math.sin(a) * rad;
        if (a === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();
      ctx.stroke();

      // Orbiting synaptic nodes & filaments
      const pts: { x: number; y: number }[] = [];
      for (const n of nodes) {
        n.angle += n.speed * speedMult;
        const nr = size * n.radiusRatio * (1 + Math.sin(t + n.angle) * 0.1);
        pts.push({
          x: cx + Math.cos(n.angle) * nr,
          y: cy + Math.sin(n.angle) * nr,
        });
      }

      ctx.strokeStyle = `rgba(${primary}, 0.24)`;
      ctx.lineWidth = 0.8;
      for (let i = 0; i < pts.length; i++) {
        const next = pts[(i + 3) % pts.length];
        ctx.beginPath();
        ctx.moveTo(pts[i].x, pts[i].y);
        ctx.lineTo(next.x, next.y);
        ctx.stroke();
      }

      for (const p of pts) {
        ctx.fillStyle = `rgba(${primary}, 0.92)`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, Math.max(1.3, size * 0.026), 0, Math.PI * 2);
        ctx.fill();
      }

      // Inner bright core nucleus
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(cx, cy, Math.max(2.2, size * 0.075 + Math.sin(t * 2) * 0.8), 0, Math.PI * 2);
      ctx.fill();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [state, size]);

  const handleInteraction = () => {
    pulseRef.current = 1.4;
    if (onClick) onClick();
  };

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size }}
      onClick={handleInteraction}
      onMouseEnter={() => {
        pulseRef.current = 0.9;
      }}
      title={title}
      className="cursor-pointer select-none shrink-0 transition-transform duration-200 hover:scale-105"
    />
  );
};
