import { FileVideo2, UploadCloud, X } from 'lucide-react'
import { useRef, useState } from 'react'

interface VideoPickerProps { onFileSelected?: (file: File | null) => void }

export function VideoPicker({ onFileSelected }: VideoPickerProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)

  function choose(next: File | null) {
    setFile(next)
    if (inputRef.current) inputRef.current.value = ''
    onFileSelected?.(next)
  }

  return (
    <div className={`upload-zone ${dragging ? 'upload-dragging' : ''} ${file ? 'upload-has-file' : ''}`}
      onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => { event.preventDefault(); setDragging(false); choose(event.dataTransfer.files.item(0)) }}>
      <input ref={inputRef} type="file" accept="video/*" hidden onChange={(event) => choose(event.target.files?.item(0) ?? null)} />
      {file ? <div className="selected-file"><div className="file-icon"><FileVideo2 size={21} /></div><div className="file-details"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(1)} MB · Ready to submit</span></div><button type="button" className="icon-button" aria-label="Remove video" onClick={() => choose(null)}><X size={17} /></button></div> : <>
        <div className="upload-icon"><UploadCloud size={24} /></div><strong>Drop your video here</strong><span>or browse files from your device</span>
        <button type="button" className="button button-secondary button-small" onClick={() => inputRef.current?.click()}>Choose video</button>
        <small>MP4, MOV, AVI or WebM · Server upload limit applies</small>
      </>}
    </div>
  )
}
