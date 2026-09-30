// Rule-based coaching feedback: compare metrics of the user and the matched
// pro at corresponding phases and turn the differences into plain-language
// notes with a severity and a drill/tip.

import { valueAt, maxIn } from './metrics.js';
import { phaseLabel } from './phases.js';

/** Average adult ratio of Neck→MidHip (2D, side view) to standing height. */
export const TORSO_TO_HEIGHT = 0.28;

function at(S, key, phase) {
  const r = Math.max(0, Math.round(S.realFps / 60));
  return valueAt(S.series[key], S.phases[phase], r);
}

/**
 * Each rule: which phase, how to read the value from a swing, tolerance, and
 * the message/tip when the user is above ("more") or below ("less") the pro.
 * `{pro}` is replaced with the pro's name.
 */
export const RULES = [
  // ---------------- Stance ----------------
  {
    id: 'stance.width', phase: 'stance', label: 'Stance width', unit: 'len', tol: 0.2, weight: 1,
    get: (S) => at(S, 'stanceWidth', 'stance'),
    more: 'Your stance is wider than {pro}’s.',
    moreTip: 'A very wide base limits your stride and load. Try bringing your feet in a couple of inches, or make sure you still get a controlled forward move.',
    less: 'Your stance is narrower than {pro}’s.',
    lessTip: 'A narrow stance needs a bigger, well-timed stride to get into a strong position. Widen your base slightly or practice a consistent stride length.',
    good: 'Stance width matches {pro}.',
  },
  {
    id: 'stance.posture', phase: 'stance', label: 'Height of stance', unit: 'len', tol: 0.15, weight: 0.8,
    get: (S) => at(S, 'posture', 'stance'),
    more: 'You stand taller than {pro} in your stance.',
    moreTip: 'Add a little more bend in the knees and hips so you start in an athletic, ready position.',
    less: 'You are more crouched than {pro} in your stance.',
    lessTip: 'A deep crouch can make it harder to track the ball and load. Try standing a bit taller.',
    good: 'Your stance height is similar to {pro}’s.',
  },
  {
    id: 'stance.weight', phase: 'stance', label: 'Weight distribution', unit: 'len', tol: 0.08, weight: 0.8,
    get: (S) => at(S, 'weightShift', 'stance'),
    more: 'Your weight starts further toward your front foot than {pro}’s.',
    moreTip: 'Set up centered or slightly on the back leg so you have something to load into and move forward from.',
    less: 'You start with more weight on your back leg than {pro}.',
    lessTip: 'That works if you still move forward on time. Make sure you don’t get stuck on your back side.',
    good: 'Your weight distribution matches {pro}.',
  },
  {
    id: 'stance.handsHeight', phase: 'stance', label: 'Hand height', unit: 'len', tol: 0.15, weight: 1,
    get: (S) => at(S, 'handsHeight', 'stance'),
    more: 'Your hands start higher than {pro}’s.',
    moreTip: 'High hands are fine if they get to the hitting slot on time. Watch for a long, loopy path down.',
    less: 'Your hands start lower than {pro}’s.',
    lessTip: 'Low hands often have to climb before they can swing (a hitch). Try setting them near the top of your back shoulder.',
    good: 'Your hand height matches {pro}.',
  },
  {
    id: 'stance.handsDepth', phase: 'stance', label: 'Hand position (front/back)', unit: 'len', tol: 0.15, weight: 1,
    get: (S) => at(S, 'handsDepth', 'stance'),
    more: 'Your hands start closer to the pitcher than {pro}’s.',
    moreTip: 'Set the hands a little further back, near the back shoulder, so the load is short and simple.',
    less: 'Your hands start further back than {pro}’s.',
    lessTip: 'Hands set far back can get stuck behind you. Keep them from wrapping behind your head.',
    good: 'Your hands are set where {pro}’s are.',
  },
  {
    id: 'stance.backElbow', phase: 'stance', label: 'Back elbow height', unit: 'len', tol: 0.15, weight: 0.6,
    get: (S) => at(S, 'backElbowHeight', 'stance'),
    more: 'Your back elbow is higher than {pro}’s.',
    moreTip: 'A high back elbow is a style choice. Just make sure it drops into the slot as the swing starts.',
    less: 'Your back elbow is lower than {pro}’s.',
    lessTip: 'A dropped back elbow at set-up can lead to a flat, pushy path. Try raising it slightly.',
    good: 'Your back elbow height matches {pro}.',
  },
  {
    id: 'stance.tilt', phase: 'stance', label: 'Upper-body lean', unit: 'deg', tol: 6, weight: 0.6,
    get: (S) => at(S, 'trunkTilt', 'stance'),
    more: 'Your upper body leans more toward the pitcher than {pro}’s.',
    moreTip: 'Stack your head over your belt buckle (or slightly back) so the load has room to move forward.',
    less: 'Your upper body leans back, away from the pitcher, more than {pro}’s.',
    lessTip: 'Leaning back at set-up can make you drift or get stuck. Stack your head over your belt buckle.',
    good: 'Your upper-body lean matches {pro}.',
  },
  // ---------------- Load ----------------
  {
    id: 'load.hands', phase: 'load', label: 'Hand load', unit: 'len', tol: 0.08, weight: 1,
    get: (S) => at(S, 'handsX', 'load') - at(S, 'handsX', 'stance'),
    more: 'Your hands move back less during the load than {pro}’s.',
    moreTip: 'Let the hands go back slightly as the front foot lifts. That stretch between hands and front side creates bat speed.',
    less: 'Your hands load further back than {pro}’s.',
    lessTip: 'A big hand load has to come all the way back. Keep it compact so you can be on time.',
    good: 'Your hands load back like {pro}’s.',
  },
  {
    id: 'load.weight', phase: 'load', label: 'Weight shift back', unit: 'len', tol: 0.07, weight: 1,
    get: (S) => at(S, 'hipTravel', 'load'),
    more: 'You shift your weight back less than {pro} during the load.',
    moreTip: 'Gather into your back hip (feel pressure on the inside of your back foot) before you stride.',
    less: 'You shift back further than {pro} during the load.',
    lessTip: 'Load into the back hip without swaying the hips past your back foot.',
    good: 'Your weight shift into the load matches {pro}.',
  },
  {
    id: 'load.legLift', phase: 'load', label: 'Leg lift', unit: 'len', tol: 0.1, weight: 0.8,
    get: (S) => maxIn(S.series.frontFootLift, S.phases.stance, S.phases.footPlant),
    more: 'Your front foot lifts higher than {pro}’s.',
    moreTip: 'A bigger leg kick needs earlier timing. Start it sooner so you are not late.',
    less: 'Your leg lift is smaller than {pro}’s.',
    lessTip: 'That is fine if you are on time. A slightly bigger gather can add rhythm and power.',
    good: 'Your leg lift is similar to {pro}’s.',
  },
  // ---------------- Foot plant ----------------
  {
    id: 'plant.stride', phase: 'footPlant', label: 'Stride length', unit: 'len', tol: 0.18, weight: 1.2,
    get: (S) => at(S, 'stride', 'footPlant'),
    more: 'Your stride is longer than {pro}’s.',
    moreTip: 'Overstriding lowers your eyes and makes it harder to rotate. Try a shorter, softer stride: step to land, not to reach.',
    less: 'Your stride is shorter than {pro}’s.',
    lessTip: 'A short stride works if you still transfer weight forward. Make sure the front foot gets down on time.',
    good: 'Your stride length matches {pro}.',
  },
  {
    id: 'plant.head', phase: 'footPlant', label: 'Head movement at foot plant', unit: 'len', tol: 0.12, weight: 1.2,
    get: (S) => at(S, 'headX', 'footPlant'),
    more: 'Your head moves toward the pitcher more than {pro}’s by foot plant.',
    moreTip: 'Keep your head quiet. Let the lower half stride while the head stays centered between your feet.',
    less: 'Your head stays further back than {pro}’s at foot plant.',
    lessTip: 'Staying back is good, but make sure your weight still moves into the front side.',
    good: 'Your head stays as quiet as {pro}’s through the stride.',
  },
  {
    id: 'plant.hands', phase: 'footPlant', label: 'Hands at foot plant', unit: 'len', tol: 0.15, weight: 1,
    get: (S) => at(S, 'handsDepth', 'footPlant'),
    more: 'Your hands have already started forward at foot plant.',
    moreTip: 'Stride forward while the hands stay back (separation). Drill: stride and pause, then swing.',
    less: 'Your hands are further back at foot plant than {pro}’s.',
    lessTip: 'Good separation. Make sure the hands still have time to get to the ball.',
    good: 'Your hands are loaded at foot plant, like {pro}’s.',
  },
  // ---------------- Contact ----------------
  {
    id: 'contact.frontLeg', phase: 'contact', label: 'Front leg brace', unit: 'deg', tol: 8, weight: 1.2,
    get: (S) => at(S, 'fKnee', 'contact'),
    more: 'Your front leg is straighter at contact than {pro}’s.',
    moreTip: 'A firm front leg is good. Make sure you are not locking out early and pulling off the ball.',
    less: 'Your front knee is more bent at contact than {pro}’s, so you brace less.',
    lessTip: 'Firm up the front leg as you rotate. A braced front side turns forward momentum into rotation.',
    good: 'Your front leg braces at contact like {pro}’s.',
  },
  {
    id: 'contact.head', phase: 'contact', label: 'Head movement to contact', unit: 'len', tol: 0.12, weight: 1.3,
    get: (S) => at(S, 'headX', 'contact'),
    more: 'Your head drifts toward the pitcher more than {pro}’s by contact.',
    moreTip: 'Head drift changes how you see the ball. Try a no-stride drill and focus on turning around a still head.',
    less: 'Your head stays further back than {pro}’s at contact.',
    lessTip: 'Staying back is fine. Just make sure you are not leaning back and uppercutting.',
    good: 'Your head stays still through contact, like {pro}’s.',
  },
  {
    id: 'contact.headDrop', phase: 'contact', label: 'Head height at contact', unit: 'len', tol: 0.1, weight: 0.8,
    get: (S) => at(S, 'headY', 'contact'),
    more: 'Your head rises more than {pro}’s by contact.',
    moreTip: 'Standing up through the swing changes your eye level. Keep your posture through contact.',
    less: 'Your head drops more than {pro}’s by contact.',
    lessTip: 'A big head drop often comes from overstriding or collapsing the back leg. Keep your eye level steady.',
    good: 'Your eye level stays steady, like {pro}’s.',
  },
  {
    id: 'contact.tilt', phase: 'contact', label: 'Spine tilt at contact', unit: 'deg', tol: 6, weight: 1,
    get: (S) => at(S, 'trunkTilt', 'contact'),
    more: 'Your upper body is more upright or forward at contact than {pro}’s, so you are less behind the ball.',
    moreTip: 'Let the back shoulder work down and under (side bend) so your head and chest stay behind the ball.',
    less: 'You tilt back, away from the pitcher, more than {pro} at contact.',
    lessTip: 'Too much tilt can cause pop-ups. Stay balanced over your base.',
    good: 'Your spine tilt at contact matches {pro}.',
  },
  {
    id: 'contact.hips', phase: 'contact', label: 'Hip turn at contact', unit: 'deg', tol: 12, weight: 1.1,
    get: (S) => at(S, 'hipTurn', 'contact'),
    more: 'Your hips are more open at contact than {pro}’s.',
    moreTip: 'Make sure the barrel keeps up. Hips that fly open too far can leave the bat dragging.',
    less: 'Your hips are less open at contact than {pro}’s.',
    lessTip: 'Lead with the hips. Fire the back hip toward the pitcher as the front foot lands (hip-lead or step-back drill).',
    good: 'Your hips open at contact like {pro}’s.',
  },
  {
    id: 'contact.shoulders', phase: 'contact', label: 'Shoulder turn at contact', unit: 'deg', tol: 12, weight: 1,
    get: (S) => at(S, 'shoulderTurn', 'contact'),
    more: 'Your shoulders are more open at contact than {pro}’s.',
    moreTip: 'Keep the front shoulder closed a little longer. The hips should lead the shoulders.',
    less: 'Your shoulders are less open at contact than {pro}’s.',
    lessTip: 'Let the torso follow the hips through. Make sure you are not blocking your rotation.',
    good: 'Your shoulder turn at contact matches {pro}.',
  },
  {
    id: 'contact.slot', phase: 'contact', label: 'Back elbow slot', unit: 'len', tol: 0.12, weight: 0.8,
    get: (S) => at(S, 'backElbowHeight', 'contact'),
    more: 'Your back elbow is higher at contact than {pro}\u2019s, so it isn\u2019t slotting down by your side.',
    moreTip: 'Let the back elbow drop into the slot, close to your back hip, as the hips turn. It shortens the path to the ball.',
    less: 'Your back elbow is lower at contact than {pro}\u2019s.',
    lessTip: 'A very low back elbow can drag the barrel under the ball. Keep the elbow slotted but let the hands stay above the ball.',
    good: 'Your back elbow slots like {pro}\u2019s at contact.',
  },
  {
    id: 'contact.point', phase: 'contact', label: 'Contact point', unit: 'len', tol: 0.15, weight: 1,
    get: (S) => at(S, 'handsToFrontHip', 'contact'),
    more: 'You make contact further out in front than {pro}.',
    moreTip: 'Contact far out front can mean you are early or lunging. Let the ball travel a little more.',
    less: 'Your contact point is deeper (closer to the catcher) than {pro}’s.',
    lessTip: 'Deep contact can mean you are late. Start your load earlier.',
    good: 'Your contact point matches {pro}.',
  },
  {
    id: 'contact.handsHeight', phase: 'contact', label: 'Hands at contact', unit: 'len', tol: 0.15, weight: 0.8,
    get: (S) => at(S, 'handsHeight', 'contact'),
    more: 'Your hands are higher at contact than {pro}’s.',
    moreTip: 'Check that you are matching the plane of the pitch and not chopping down.',
    less: 'Your hands are lower at contact than {pro}’s.',
    lessTip: 'Dropping the hands can create a long, uphill path. Keep your hands above the ball.',
    good: 'Your hand height at contact matches {pro}.',
  },
  {
    id: 'contact.shoulderDrop', phase: 'contact', label: 'Back shoulder drop', unit: 'len', tol: 0.1, weight: 0.7,
    get: (S) => at(S, 'shoulderDrop', 'contact'),
    more: 'Your back shoulder drops more than {pro}’s at contact.',
    moreTip: 'Too much drop creates an uppercut. Level it off slightly.',
    less: 'Your back shoulder does not drop as much as {pro}’s at contact.',
    lessTip: 'Let the back shoulder tilt down to match the plane of the pitch.',
    good: 'Your shoulder tilt at contact matches {pro}.',
  },
  // ---------------- Extension ----------------
  {
    id: 'ext.reach', phase: 'extension', label: 'Extension toward the pitcher', unit: 'len', tol: 0.15, weight: 0.9,
    get: (S) => at(S, 'extensionX', 'extension'),
    more: 'Your hands reach further out toward the pitcher after contact than {pro}\u2019s.',
    moreTip: 'Good extension. Make sure it isn\u2019t coming from lunging forward with the upper body.',
    less: 'Your hands don\u2019t extend out toward the pitcher as far as {pro}\u2019s after contact.',
    lessTip: 'Drive the hands through the ball toward the pitcher and stay long through the zone before letting the arms fold.',
    good: 'Your extension after contact matches {pro}.',
  },
  // ---------------- Finish ----------------
  {
    id: 'finish.balance', phase: 'finish', label: 'Finish balance', unit: 'len', tol: 0.2, weight: 0.9,
    get: (S) => at(S, 'headOverFeet', 'finish'),
    more: 'You finish with your head further toward the pitcher than {pro}, so you may be falling forward.',
    moreTip: 'Finish balanced over a firm front side. Hold your finish for two seconds after every rep.',
    less: 'You finish leaning back more than {pro}.',
    lessTip: 'Falling back at the finish can come from pulling off the ball. Finish tall over your front leg.',
    good: 'You finish balanced, like {pro}.',
  },
  {
    id: 'finish.hands', phase: 'finish', label: 'Finish height', unit: 'len', tol: 0.25, weight: 0.6,
    get: (S) => at(S, 'handsHeight', 'finish'),
    more: 'Your finish is higher than {pro}’s.',
    moreTip: 'A high finish is fine if it comes from a good path. Make sure you are not lifting early.',
    less: 'Your finish is lower than {pro}’s.',
    lessTip: 'Let the bat keep going and finish high around the front shoulder.',
    good: 'Your finish height matches {pro}.',
  },
  {
    id: 'finish.rotation', phase: 'finish', label: 'Shoulder rotation at finish', unit: 'deg', tol: 20, weight: 0.7,
    get: (S) => at(S, 'shoulderTurn', 'finish'),
    more: 'Your shoulders rotate further through the finish than {pro}’s.',
    moreTip: 'Full rotation is good. Make sure you keep your balance.',
    less: 'Your shoulders do not rotate as far through the finish as {pro}’s.',
    lessTip: 'Let the back shoulder come all the way through so your chest faces the pitcher at the finish.',
    good: 'Your finish rotation matches {pro}.',
  },
  // ---------------- Timing ----------------
  {
    id: 'timing.stride', phase: 'footPlant', label: 'Stride timing (load → foot plant)', unit: 'sec', relTol: 0.25, weight: 0.8, timing: true,
    get: (S) => (S.phases.footPlant - S.phases.load) / S.realFps,
    more: 'Your stride (load to foot plant) is slower than {pro}’s.',
    moreTip: 'A slow stride is fine if you start early. Use it to time the pitcher.',
    less: 'Your stride (load to foot plant) is quicker than {pro}’s.',
    lessTip: 'A quick stride can rush you. Try a slower, controlled gather.',
    good: 'Your stride timing matches {pro}.',
  },
  {
    id: 'timing.swing', phase: 'contact', label: 'Swing time (foot plant → contact)', unit: 'sec', relTol: 0.2, weight: 1.2, timing: true,
    get: (S) => (S.phases.contact - S.phases.footPlant) / S.realFps,
    more: 'It takes you longer to get from foot plant to contact than {pro}.',
    moreTip: 'Shorten the path and let the hips start the swing. Connection and bat-speed drills help (overload/underload bats, short-bat drill).',
    less: 'You get from foot plant to contact faster than {pro}.',
    lessTip: 'Quick to the ball. Keep it up.',
    good: 'Your swing time matches {pro}.',
  },
];

/** Sequence check: do the hips lead the shoulders at contact like the pro? */
function sequenceItem(user, pro, proName) {
  const uh = at(user, 'hipTurn', 'contact');
  const us = at(user, 'shoulderTurn', 'contact');
  const ph = at(pro, 'hipTurn', 'contact');
  const ps = at(pro, 'shoulderTurn', 'contact');
  if (![uh, us, ph, ps].every(Number.isFinite)) return null;
  const userLead = uh - us;
  const proLead = ph - ps;
  const bad = userLead < -5 && proLead >= 0;
  return {
    id: 'contact.sequence',
    phase: 'contact',
    label: 'Hips lead shoulders',
    unit: 'deg',
    user: userLead,
    pro: proLead,
    delta: userLead - proLead,
    tol: 12,
    severity: bad ? 'major' : userLead < proLead - 15 ? 'minor' : 'good',
    score: bad ? 2.5 : Math.abs(userLead - proLead) / 15,
    weight: 1.2,
    message: bad
      ? 'Your shoulders are turning ahead of your hips.'
      : userLead < proLead - 15
        ? `Your hips lead your shoulders less than ${proName}’s do.`
        : `Your hips lead your shoulders at contact, like ${proName}’s.`,
    tip: bad || userLead < proLead - 15
      ? 'Power flows hips, then torso, then arms, then bat. Start the swing with the back hip while the front shoulder stays closed (hip-lead drill).'
      : '',
  };
}

/**
 * Evaluate every rule.
 * @param {{series:object, phases:object, realFps:number}} user swing (phases from the alignment)
 * @param {{series:object, phases:object, realFps:number}} pro
 * @param {{proName:string, timingKnown?:boolean}} opts
 */
export function evaluateFeedback(user, pro, { proName, timingKnown = true } = {}) {
  const items = [];
  for (const rule of RULES) {
    if (rule.timing && !timingKnown) continue;
    const u = rule.get(user);
    const p = rule.get(pro);
    if (!Number.isFinite(u) || !Number.isFinite(p)) continue;
    const tol = rule.relTol ? Math.max(0.03, Math.abs(p) * rule.relTol) : rule.tol;
    const d = u - p;
    const ratio = Math.abs(d) / tol;
    const severity = ratio <= 1 ? 'good' : ratio <= 2 ? 'minor' : 'major';
    const dir = d > 0 ? 'more' : 'less';
    const fill = (s) => (s || '').replaceAll('{pro}', proName);
    items.push({
      id: rule.id,
      phase: rule.phase,
      label: rule.label,
      unit: rule.unit,
      user: u,
      pro: p,
      delta: d,
      tol,
      severity,
      score: ratio,
      weight: rule.weight,
      message: fill(severity === 'good' ? rule.good : rule[dir]),
      tip: severity === 'good' ? '' : fill(rule[`${dir}Tip`]),
    });
  }
  const seq = sequenceItem(user, pro, proName);
  if (seq) items.push(seq);
  return items;
}

/** Top priorities (worst weighted deviations) and strengths. */
export function summarize(items, n = 3) {
  const issues = items
    .filter((i) => i.severity !== 'good')
    .sort((a, b) => b.score * b.weight - a.score * a.weight);
  const strengths = items.filter((i) => i.severity === 'good').sort((a, b) => b.weight - a.weight);
  return { priorities: issues.slice(0, n), issues, strengths };
}

/** Format a value for display. `len` values are converted to inches. */
export function formatValue(v, unit, torsoIn = 20) {
  if (!Number.isFinite(v)) return '—';
  switch (unit) {
    case 'len':
      return `${Math.round(v * torsoIn)} in`;
    case 'deg':
      return `${Math.round(v)}°`;
    case 'sec':
      return `${v.toFixed(2)} s`;
    default:
      return String(Math.round(v * 100) / 100);
  }
}

export function formatDelta(v, unit, torsoIn = 20) {
  if (!Number.isFinite(v)) return '';
  const s = formatValue(Math.abs(v), unit, torsoIn);
  if (parseFloat(s) === 0) return s; // rounds to zero: no sign
  return `${v >= 0 ? '+' : '−'}${s}`;
}

export { phaseLabel };
