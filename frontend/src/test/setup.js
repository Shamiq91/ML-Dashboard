// jsdom replaces Node's fetch/FormData/Blob/File; restore a matching Node/undici set so
// uploads from the app reach the real backend.
import { Blob as NBlob, File as NFile } from 'node:buffer'
import { fetch as ufetch, FormData as UFormData, WebSocket as UWebSocket } from 'undici'

globalThis.fetch = ufetch
globalThis.WebSocket = UWebSocket
globalThis.FormData = UFormData
globalThis.Blob = NBlob
globalThis.File = NFile
globalThis.ResizeObserver = class {
  observe() {}
  unobserve() {}
  disconnect() {}
}
if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {}
