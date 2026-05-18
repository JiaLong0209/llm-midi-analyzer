import * as Tone from "tone";
import { createWaveRollPlayer } from "@/lib/components/player/wave-roll/player";

/**
 * WaveRoll Web Component
 *
 * Usage:
 * <wave-roll
 *   files='[{"path": "file.mid", "name": "File Name", "type": "midi"}]'
 *   style="width: 100%
 * ></wave-roll>
 *
 * Note:
 * - The component accepts `name` for user-facing labels.
 */
class WaveRollElement extends HTMLElement {
  private player: any = null;
  private container: HTMLDivElement | null = null;

  constructor() {
    super();
    this.attachShadow({ mode: 'open' });
  }

  connectedCallback() {
    this.render();
    this.initializePlayer();
    this.setupAudioResume();
  }

  private setupAudioResume() {
    const resumeAudio = async () => {
      try {
        // Synchronous activation inside the user gesture event handler call stack
        await Tone.start();
        const ctx = Tone.getContext();
        if (ctx && ctx.rawContext && ctx.rawContext.state === 'suspended') {
          await ctx.rawContext.resume();
        }
        console.log('[WaveRollElement] AudioContext successfully resumed synchronously!');
      } catch (e) {
        console.warn('[WaveRollElement] AudioContext resume failed:', e);
      }
      // Remove listeners once context is running
      document.removeEventListener('click', resumeAudio);
      document.removeEventListener('pointerdown', resumeAudio);
      document.removeEventListener('keydown', resumeAudio);
    };
    document.addEventListener('click', resumeAudio);
    document.addEventListener('pointerdown', resumeAudio);
    document.addEventListener('keydown', resumeAudio);
  }

  disconnectedCallback() {
    if (this.player && typeof this.player.destroy === 'function') {
      this.player.destroy();
    }
  }

  static get observedAttributes() {
    return ['files', 'readonly', 'config'];
  }

  attributeChangedCallback(name: string, oldValue: string, newValue: string) {
    if (name === 'files' && oldValue !== newValue) {
      this.initializePlayer();
      return;
    }
    if (name === 'readonly' && this.player && oldValue !== newValue) {
      const ro = typeof (this as any).hasAttribute === 'function' ? this.hasAttribute('readonly') : !!newValue;
      try {
        this.player.setPermissions?.({ canAddFiles: !ro, canRemoveFiles: !ro });
      } catch {}
      return;
    }
    if (name === 'config' && oldValue !== newValue) {
      this.applyConfig();
      return;
    }
  }

  private applyConfig() {
    const configAttr = this.getAttribute('config');
    if (!configAttr || !this.player) return;
    try {
      const config = JSON.parse(configAttr);
      const pianoRoll = this.player.pianoRollManager?._instance || this.player.visualizationEngine?.getPianoRollInstance()?._instance;
      if (pianoRoll) {
        if (config.noteColor !== undefined) {
          pianoRoll.options.noteColor = typeof config.noteColor === 'string' ? parseInt(config.noteColor.replace('#', ''), 16) : config.noteColor;
        }
        if (config.backgroundColor !== undefined) {
          const bgNum = typeof config.backgroundColor === 'string' ? parseInt(config.backgroundColor.replace('#', ''), 16) : config.backgroundColor;
          pianoRoll.options.backgroundColor = bgNum;
          if (pianoRoll.app && pianoRoll.app.renderer) {
            pianoRoll.app.renderer.background.color = bgNum;
          }
        }
        if (config.timeLabelFontSize !== undefined) {
          pianoRoll.options.timeLabelFontSize = Number(config.timeLabelFontSize);
        }
        if (config.zoomY !== undefined) {
          pianoRoll.state.zoomY = Number(config.zoomY);
        }
        if (config.showTimeGrid !== undefined) {
          pianoRoll.options.showTimeGrid = !!config.showTimeGrid;
        }
        if (config.globalTransitionEnabled !== undefined) {
          pianoRoll.options.globalTransitionEnabled = !!config.globalTransitionEnabled;
        }
        if (config.transitionDuration !== undefined) {
          pianoRoll.options.transitionDuration = Number(config.transitionDuration);
        }
        if (config.tempo !== undefined) {
          pianoRoll.options.tempo = Number(config.tempo);
        }
        if (config.colorByTrack !== undefined) {
          pianoRoll.options.colorByTrack = !!config.colorByTrack;
        }
        if (config.tempoMap !== undefined) {
          pianoRoll.options.tempoMap = config.tempoMap;
        }
        if (config.beatsPerBar !== undefined) {
          pianoRoll.options.beatsPerBar = Number(config.beatsPerBar);
        }
        if (config.gridSubdivision !== undefined) {
          pianoRoll.options.gridSubdivision = Number(config.gridSubdivision);
        }
        pianoRoll.needsNotesRedraw = true;
        pianoRoll.requestRender();
      }
    } catch (e) {
      console.error('Failed to apply config:', e);
    }
  }

  private render() {
    if (!this.shadowRoot) return;

    // Create styles
    const style = document.createElement('style');
    style.textContent = `
      :host {
        display: block;
        width: 100%;
        height: 100%;
        overflow-y: auto;
      }
      .wave-roll-container {
        width: 100%;
        height: 100%;
        position: relative;
      }
    `;

    // Create container
    this.container = document.createElement('div');
    this.container.className = 'wave-roll-container';

    // Clear and append
    this.shadowRoot.innerHTML = '';
    this.shadowRoot.appendChild(style);
    this.shadowRoot.appendChild(this.container);
  }

  private async initializePlayer() {
    if (!this.container) return;

    // Clean up existing player
    if (this.player) {
      if (typeof this.player.destroy === 'function') {
        this.player.destroy();
      }
      this.player = null;
    }

    // Parse files attribute
    const filesAttr = this.getAttribute('files');
    
    let files = [] as any[];
    if (filesAttr) {
      try {
        files = JSON.parse(filesAttr);
      } catch (e) {
        console.error('Invalid files attribute:', e);
        return;
      }
    }

    // Always create the player, even with no files (shows empty state)
    try {
      // Normalize incoming items: use `name` as the primary property
      const normalized = (Array.isArray(files) ? files : []).map((f: any) => {
        const mapped: any = { path: f.path };
        if (typeof f?.name === 'string') {
          mapped.name = f.name;
        }
        if (f && typeof f.type === 'string') {
          mapped.type = f.type;
        }
        if (f && typeof f.color !== 'undefined') {
          mapped.color = f.color;
        }
        return mapped;
      });
      // Parse config attribute for initial settings
      const configAttr = this.getAttribute('config');
      let initialPianoRollConfig = {} as any;
      if (configAttr) {
        try {
          const parsed = JSON.parse(configAttr);
          if (parsed.noteColor !== undefined) {
            initialPianoRollConfig.noteColor = typeof parsed.noteColor === 'string' ? parseInt(parsed.noteColor.replace('#', ''), 16) : parsed.noteColor;
          }
          if (parsed.backgroundColor !== undefined) {
            initialPianoRollConfig.backgroundColor = typeof parsed.backgroundColor === 'string' ? parseInt(parsed.backgroundColor.replace('#', ''), 16) : parsed.backgroundColor;
          }
          if (parsed.timeLabelFontSize !== undefined) {
            initialPianoRollConfig.timeLabelFontSize = Number(parsed.timeLabelFontSize);
          }
          if (parsed.zoomY !== undefined) {
            initialPianoRollConfig.defaultZoomY = Number(parsed.zoomY);
          }
          if (parsed.showTimeGrid !== undefined) {
            initialPianoRollConfig.showTimeGrid = !!parsed.showTimeGrid;
          }
          if (parsed.globalTransitionEnabled !== undefined) {
            initialPianoRollConfig.globalTransitionEnabled = !!parsed.globalTransitionEnabled;
          }
          if (parsed.transitionDuration !== undefined) {
            initialPianoRollConfig.transitionDuration = Number(parsed.transitionDuration);
          }
          if (parsed.tempo !== undefined) {
            initialPianoRollConfig.tempo = Number(parsed.tempo);
          }
          if (parsed.colorByTrack !== undefined) {
            initialPianoRollConfig.colorByTrack = !!parsed.colorByTrack;
          }
          if (parsed.tempoMap !== undefined) {
            initialPianoRollConfig.tempoMap = parsed.tempoMap;
          }
          if (parsed.beatsPerBar !== undefined) {
            initialPianoRollConfig.beatsPerBar = Number(parsed.beatsPerBar);
          }
          if (parsed.gridSubdivision !== undefined) {
            initialPianoRollConfig.gridSubdivision = Number(parsed.gridSubdivision);
          }
        } catch {}
      }

      this.player = await createWaveRollPlayer(this.container, normalized, {
        pianoRoll: {
          showWaveformBand: false,
          ...initialPianoRollConfig
        }
      });
      // Apply readonly after player creation if attribute present
      if (typeof (this as any).hasAttribute === 'function' ? this.hasAttribute('readonly') : true) {
        // If hasAttribute is unavailable (test stubs), treat presence of attribute string as truthy
        this.player.setPermissions?.({ canAddFiles: false, canRemoveFiles: false });
      }
      // Notify listeners the component has finished initialization
      this.dispatchEvent(new Event('load'));
    } catch (e) {
      console.error('Failed to initialize WaveRoll player:', e);
    }
  }

  // Expose minimal control API for tests/integration
  public async play(): Promise<void> {
    if (this.player?.play) {
      await this.player.play();
    }
  }

  public pause(): void {
    if (this.player?.pause) {
      this.player.pause();
    }
  }

  public get isPlaying(): boolean {
    return !!this.player?.isPlaying;
  }

  /**
   * Seek to a specific time (seconds).
   * Provided for E2E/manual testing via index.html.
   */
  public seek(time: number): void {
    try {
      // Prefer direct method on player if available
      if (typeof this.player?.seek === 'function') {
        this.player.seek(time);
        return;
      }
      // Fallback: try visualization engine behind the player
      (this.player as any)?.visualizationEngine?.seek?.(time, true);
    } catch (e) {
      console.error('WaveRollElement.seek failed:', e);
    }
  }

  /**
   * Return lightweight state for assertions in tests.
   */
  public getState(): any {
    try {
      if (typeof this.player?.getState === 'function') {
        return this.player.getState();
      }
      return (this.player as any)?.visualizationEngine?.getState?.();
    } catch (e) {
      console.error('WaveRollElement.getState failed:', e);
      return null;
    }
  }
}

// Register the custom element
if (!customElements.get('wave-roll')) {
  customElements.define('wave-roll', WaveRollElement);
}

export { WaveRollElement };
