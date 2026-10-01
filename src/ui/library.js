// Saved swings: analyzed poses (and, when there is room, the video itself)
// kept in this browser's IndexedDB so the same swing doesn't have to be
// uploaded and analyzed again. Nothing leaves the device.

const DB_NAME = 'swing-match';
const DB_VERSION = 1;
const META = 'swings'; // small records: analysis + settings + thumbnail
const VIDEOS = 'videos'; // { id, blob } kept apart so listing stays fast

let dbPromise = null;

function openDb() {
  if (!dbPromise) {
    dbPromise = new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') {
        reject(new Error('IndexedDB is not available'));
        return;
      }
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(META)) db.createObjectStore(META, { keyPath: 'id' });
        if (!db.objectStoreNames.contains(VIDEOS)) db.createObjectStore(VIDEOS, { keyPath: 'id' });
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
      req.onblocked = () => reject(new Error('Saved swings are open in another tab that needs reloading'));
    });
    dbPromise.catch(() => {
      dbPromise = null;
    });
  }
  return dbPromise;
}

function run(store, mode, fn) {
  return openDb().then(
    (db) =>
      new Promise((resolve, reject) => {
        const tx = db.transaction(store, mode);
        let result;
        Promise.resolve(fn(tx.objectStore(store))).then((r) => {
          result = r;
        });
        tx.oncomplete = () => resolve(result);
        tx.onerror = () => reject(tx.error);
        tx.onabort = () => reject(tx.error || new Error('Transaction aborted'));
      }),
  );
}

function req2promise(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function isAvailable() {
  try {
    await openDb();
    return true;
  } catch {
    return false;
  }
}

/** All saved swings, newest first (without video blobs). */
export async function listSwings() {
  const all = await run(META, 'readonly', (s) => req2promise(s.getAll()));
  return (all || []).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
}

export function getSwing(id) {
  return run(META, 'readonly', (s) => req2promise(s.get(id)));
}

export async function putSwing(record) {
  record.updatedAt = Date.now();
  await run(META, 'readwrite', (s) => s.put(record));
  return record;
}

/** Merge fields into a saved swing (no-op if it was deleted). */
export async function updateSwing(id, patch) {
  const rec = await getSwing(id);
  if (!rec) return null;
  return putSwing({ ...rec, ...patch });
}

/**
 * Store the video file. Returns false (and stores nothing) when the browser
 * is out of space; the swing is still saved without its video.
 */
export async function putVideo(id, blob) {
  try {
    await run(VIDEOS, 'readwrite', (s) => s.put({ id, blob }));
    return true;
  } catch (e) {
    console.warn('Could not store the video:', e);
    return false;
  }
}

export async function getVideo(id) {
  const rec = await run(VIDEOS, 'readonly', (s) => req2promise(s.get(id)));
  return rec?.blob || null;
}

export async function deleteSwing(id) {
  await run(META, 'readwrite', (s) => s.delete(id));
  await run(VIDEOS, 'readwrite', (s) => s.delete(id));
}

/** Ask the browser not to evict saved swings under storage pressure. */
export function requestPersistence() {
  try {
    navigator.storage?.persist?.();
  } catch {
    /* not supported */
  }
}

/** Small JPEG of the current video frame (or null). */
export function thumbnail(video, width = 160) {
  try {
    if (!video?.videoWidth) return null;
    const h = Math.round((width * video.videoHeight) / video.videoWidth);
    const c = document.createElement('canvas');
    c.width = width;
    c.height = h;
    c.getContext('2d').drawImage(video, 0, 0, width, h);
    return c.toDataURL('image/jpeg', 0.7);
  } catch {
    return null;
  }
}
