export const QUESTIONS = Object.freeze([
  { id: 1, text: "What's your name?", color: '#c92a2a' },
  { id: 2, text: 'Where are you from?', color: '#c2410c' },
  { id: 3, text: 'How old are you?', color: '#4f46e5' },
  { id: 4, text: "What's the first letter of your name?", color: '#087f5b' },
  { id: 5, text: "What's your favourite number?", color: '#a65a00' },
]);

export function normalizeDegrees(angle) {
  return ((angle % 360) + 360) % 360;
}

export function targetRotation(currentRotation, selectedIndex, count, turns = 5) {
  if (count < 1 || selectedIndex < 0 || selectedIndex >= count) {
    throw new RangeError('Selected segment must be on the wheel.');
  }
  const step = 360 / count;
  const target = normalizeDegrees(-selectedIndex * step);
  const delta = normalizeDegrees(target - normalizeDegrees(currentRotation));
  return currentRotation + turns * 360 + delta;
}

export function indexAtPointer(rotation, count) {
  if (count < 1) return -1;
  const step = 360 / count;
  return Math.round(normalizeDegrees(-rotation) / step) % count;
}

export function pointAt(angle, radius, center = 300) {
  const radians = angle * Math.PI / 180;
  return {
    x: center + radius * Math.cos(radians),
    y: center + radius * Math.sin(radians),
  };
}

export function segmentPath(index, count, radius = 282, center = 300) {
  if (count < 2 || index < 0 || index >= count) {
    throw new RangeError('A segment path needs at least two sectors.');
  }
  const step = 360 / count;
  const mid = 180 + index * step;
  const start = pointAt(mid - step / 2, radius, center);
  const end = pointAt(mid + step / 2, radius, center);
  const largeArc = step > 180 ? 1 : 0;
  return `M ${center} ${center} L ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y} Z`;
}

export function wrapQuestion(text, maxLength = 16) {
  const lines = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? `${line} ${word}` : word;
    if (line && next.length > maxLength) {
      lines.push(line);
      line = word;
    } else {
      line = next;
    }
  }
  if (line) lines.push(line);
  return lines;
}

export function randomIndex(count) {
  if (count < 1) throw new RangeError('Cannot select from an empty wheel.');
  if (!globalThis.crypto?.getRandomValues) return Math.floor(Math.random() * count);
  const range = 0x100000000;
  const limit = Math.floor(range / count) * count;
  const value = new Uint32Array(1);
  do { globalThis.crypto.getRandomValues(value); } while (value[0] >= limit);
  return value[0] % count;
}
