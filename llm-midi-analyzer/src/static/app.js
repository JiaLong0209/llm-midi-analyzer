document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const btnPianoRoll = document.getElementById('btn-piano-roll');
    const btnGraphRag = document.getElementById('btn-graph-rag');
    const pianoRollView = document.getElementById('piano-roll-view');
    const graphRagView = document.getElementById('graph-rag-view');
    const btnToggleChat = document.getElementById('btn-toggle-chat');
    const sidebar = document.getElementById('sidebar');
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const midiUpload = document.getElementById('midi-upload');
    const uploadStatus = document.getElementById('upload-status');
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnSend = document.getElementById('btn-send');
    const chatHistory = document.getElementById('chat-history');
    const midiSampleList = document.getElementById('midi-sample-list');
    const btnRefreshSamples = document.getElementById('btn-refresh-samples');
    
    const playerWrapper = document.querySelector('.player-wrapper');
    const playerEmptyState = document.getElementById('player-empty-state');
    const mainVisualizer = document.getElementById('main-visualizer');
    const staffVisualizer = document.getElementById('staff-visualizer');
    const mainPlayer = document.getElementById('main-player');
    
    let lastUploadedFile = null;
    const waveRoll = document.getElementById('waveroll-visualizer');
    const toggleWaveRoll = document.getElementById('toggle-waveroll');
    const rollContainer = document.getElementById('roll-container');
    const staffContainer = document.getElementById('staff-container');
    
    const networkContainer = document.getElementById('network-container');
    const graphEmptyState = document.getElementById('graph-empty-state');
    
    let selectedFile = null;
    let currentSessionId = null;
    
    // Graph state
    let nodes = null;
    let edges = null;
    let network = null;
    const rawNodesMap = {};

    // --- Left Sidebar Resizer dragging ---
    const sidebarLeft = document.getElementById('sidebar-left');
    const resizerLeft = document.getElementById('left-sidebar-resizer');
    
    if (sidebarLeft && resizerLeft) {
        let isResizing = false;
        
        // Load stored width if exists
        const storedWidth = localStorage.getItem('sidebarWidth');
        if (storedWidth) {
            sidebarLeft.style.width = `${storedWidth}px`;
        }

        resizerLeft.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            resizerLeft.classList.add('dragging');
            e.preventDefault();
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const newWidth = Math.max(180, Math.min(450, e.clientX));
            sidebarLeft.style.width = `${newWidth}px`;
            localStorage.setItem('sidebarWidth', newWidth);
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = 'default';
                resizerLeft.classList.remove('dragging');
            }
        });
    }

    // --- KaTeX rendering helper ---
    function renderLatex(element) {
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true },
                ],
                throwOnError: false
            });
        }
    }

    // --- Sidebar Resizing ---
    const resizer = document.getElementById('sidebar-resizer');
    let isResizing = false;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        resizer.classList.add('resizing');
        document.body.style.cursor = 'col-resize';
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        const newWidth = document.body.clientWidth - e.clientX;
        if (newWidth >= 300 && newWidth <= 800) {
            sidebar.style.width = `${newWidth}px`;
            const main = document.querySelector('.main-view');
            if (!sidebar.classList.contains('closed')) {
                // main.style.marginRight = `${newWidth}px`; // Optional if position absolute, but flex handles it
            }
        }
    });

    document.addEventListener('mouseup', () => {
        isResizing = false;
        resizer.classList.remove('resizing');
        document.body.style.cursor = 'default';
    });

    // --- View Toggling ---
    btnPianoRoll.addEventListener('click', () => {
        btnPianoRoll.classList.add('active');
        btnGraphRag.classList.remove('active');
        pianoRollView.classList.add('active');
        graphRagView.classList.remove('active');
    });

    btnGraphRag.addEventListener('click', () => {
        btnGraphRag.classList.add('active');
        btnPianoRoll.classList.remove('active');
        graphRagView.classList.add('active');
        pianoRollView.classList.remove('active');
        if (network) {
            network.fit(); // Recenter graph when showing
        }
    });

    btnToggleChat.addEventListener('click', () => {
        sidebar.classList.toggle('closed');
        resizer.style.display = sidebar.classList.contains('closed') ? 'none' : 'block';
    });

    // --- Sidebar Tabs ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // --- MIDI Visualizer Patching ---
    // Instead of hacking redraw, we set a custom visualizer object on the player
    mainPlayer.visualizer = {
        redraw: (note, active) => {
            if (mainVisualizer.redraw) mainVisualizer.redraw(note, active);
            if (staffVisualizer.redraw) staffVisualizer.redraw(note, active);
        },
        clear: () => {
            if (mainVisualizer.clear) mainVisualizer.clear();
            if (staffVisualizer.clear) staffVisualizer.clear();
        }
    };

    // --- File Upload & Handling ---
    function handleMidiFile(file) {
        selectedFile = file;
        lastUploadedFile = file;
        uploadStatus.textContent = file.name;
        uploadStatus.style.color = '#60a5fa';
        
        // Generate local URL for immediate playback
        const objectUrl = URL.createObjectURL(file);
        
        // Update Magenta
        mainVisualizer.setAttribute('src', objectUrl);
        mainPlayer.setAttribute('src', objectUrl);
        staffVisualizer.setAttribute('src', objectUrl);
        
        // Update WaveRoll
        if (waveRoll) {
            waveRoll.setAttribute('files', JSON.stringify([
                { "path": objectUrl, "name": file.name, "type": "midi" }
            ]));
        }
        
        playerEmptyState.style.display = 'none';
        playerWrapper.style.display = 'flex';

        // Show both Send and Analyze buttons side by side
        btnSend.style.display = 'inline-flex';
        btnAnalyze.style.display = 'inline-flex';

        appendMessage('user', `Loaded MIDI: **${file.name}**\n\nClick **Analyze** to start analysis.`);
    }

    // --- Visualizer Switching Logic ---
    function updateVisualizerMode() {
        const useWaveRoll = toggleWaveRoll?.checked;
        if (useWaveRoll) {
            mainVisualizer.style.display = 'none';
            waveRoll.style.display = 'flex';
        } else {
            mainVisualizer.style.display = 'flex';
            waveRoll.style.display = 'none';
        }
        localStorage.setItem('useWaveRoll', useWaveRoll);
    }

    toggleWaveRoll?.addEventListener('change', updateVisualizerMode);

    // --- Sync WaveRoll with Master Player (Magenta) ---
    let syncAnimationFrameId = null;
    function syncLoop() {
        if (mainPlayer.playing && toggleWaveRoll?.checked && waveRoll) {
            waveRoll.seek(mainPlayer.currentTime);
        }
        if (mainPlayer.playing) {
            syncAnimationFrameId = requestAnimationFrame(syncLoop);
        }
    }
    function startSync() {
        if (syncAnimationFrameId) cancelAnimationFrame(syncAnimationFrameId);
        syncAnimationFrameId = requestAnimationFrame(syncLoop);
    }
    function stopSync() {
        if (syncAnimationFrameId) cancelAnimationFrame(syncAnimationFrameId);
    }

    mainPlayer.addEventListener('start', () => {
        // Do not call waveRoll.play() so internal timer doesn't compete with external seek updates
        startSync();
    });
    mainPlayer.addEventListener('stop', () => {
        if (waveRoll) waveRoll.pause();
        stopSync();
    });

    // Initial state from localStorage (default to true if not set)
    const storedPref = localStorage.getItem('useWaveRoll');
    const defaultUseWaveRoll = storedPref === null ? true : storedPref === 'true';
    if (toggleWaveRoll) {
        toggleWaveRoll.checked = defaultUseWaveRoll;
        updateVisualizerMode();
    }

    midiUpload.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleMidiFile(e.target.files[0]);
        } else {
            selectedFile = null;
            uploadStatus.textContent = '';
            btnSend.style.display = 'inline-flex';
            btnAnalyze.style.display = 'none';
        }
    });

    // --- Analysis Pipeline ---
    btnAnalyze.addEventListener('click', async () => {
        if (!selectedFile && lastUploadedFile) {
            selectedFile = lastUploadedFile;
        }
        if (!selectedFile) {
            alert('Please select a MIDI file first.');
            return;
        }

        const formData = new FormData();
        formData.append('file', selectedFile);

        const message = chatInput?.value?.trim() || '';
        if (message) {
            formData.append('user_prompt', message);
            appendMessage('user', `Re-analyzing with prompt: "${message}"`);
            chatInput.value = '';
        }

        btnAnalyze.disabled = true;
        btnAnalyze.textContent = 'Analyzing...';
        appendMessage('bot', 'Initiating Multimodal Analysis Pipeline... <br><small>Loading Heavy Models...</small>');
        
        // Pass all settings to backend
        const modelName = document.getElementById('cfg-model')?.value || 'gemini-3.1-flash-lite';
        formData.append('model_name', modelName);
        formData.append('temperature', document.getElementById('cfg-temperature')?.value || '0.3');
        formData.append('enable_music21', document.getElementById('cfg-m21')?.checked ? 'true' : 'false');
        formData.append('include_detailed_tracks', document.getElementById('cfg-detailed-tracks')?.checked ? 'true' : 'false');
        formData.append('enable_rag', document.getElementById('cfg-rag')?.checked ? 'true' : 'false');
        formData.append('enable_cag', document.getElementById('cfg-cag')?.checked ? 'true' : 'false');

        const startBar = document.getElementById('cfg-start-bar')?.value;
        const endBar = document.getElementById('cfg-end-bar')?.value;
        if (startBar) formData.append('start_measure', startBar);
        if (endBar) formData.append('end_measure', endBar);
        
        let ragLang = document.getElementById('cfg-rag-lang')?.value || 'zh-tw';
        if (ragLang === 'custom') {
            ragLang = document.getElementById('cfg-rag-lang-custom')?.value || 'zh-tw';
        }
        formData.append('rag_lang', ragLang);

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Network response was not ok');
            
            const result = await response.json();
            
            if (result.session_id) {
                currentSessionId = result.session_id;
            }

            // Auto-update WaveRoll tempo and dynamic tempoMap
            if (result.music21_data && waveRoll) {
                const m21 = result.music21_data;
                const configAttr = waveRoll.getAttribute('config');
                let currentCfg = {};
                if (configAttr) {
                    try { currentCfg = JSON.parse(configAttr); } catch {}
                }
                
                if (m21.bpm && m21.bpm !== 'Unknown') {
                    currentCfg.tempo = Number(m21.bpm);
                    const wrTempoInput = document.getElementById('cfg-waveroll-tempo');
                    const wrTempoVal = document.getElementById('waveroll-tempo-val');
                    if (wrTempoInput) {
                        wrTempoInput.value = Math.round(Number(m21.bpm));
                        if (wrTempoVal) wrTempoVal.textContent = Math.round(Number(m21.bpm)) + ' BPM';
                    }
                }
                if (m21.tempo_map) {
                    currentCfg.tempoMap = m21.tempo_map;
                }
                if (m21.time_signature && typeof m21.time_signature === 'string') {
                    const parts = m21.time_signature.split('/');
                    if (parts.length > 0) {
                        const num = parseInt(parts[0], 10);
                        if (!isNaN(num) && num > 0) {
                            currentCfg.beatsPerBar = num;
                            const wrBeatsSelect = document.getElementById('cfg-waveroll-beats-per-bar');
                            if (wrBeatsSelect) {
                                wrBeatsSelect.value = num.toString();
                            }
                        }
                    }
                }
                
                waveRoll.setAttribute('config', JSON.stringify(currentCfg));
            }

            appendMessage('bot', '✅ Analysis Complete! Rendering Report...');
            
            // Render Markdown using marked.js
            const mdHtml = marked.parse(result.final_report);
            appendMessage('bot', mdHtml);

            // Render Graph
            if (result.graph_data && result.graph_data.nodes.length > 0) {
                renderGraph(result.graph_data);
                graphEmptyState.style.display = 'none';
                networkContainer.style.display = 'block';
            }

        } catch (error) {
            console.error('Analysis failed:', error);
            appendMessage('bot', `❌ Analysis Failed: ${error.message}`);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = 'Analyze';
            // Keep both buttons visible so the user can easily re-analyze with a new prompt or range
            btnSend.style.display = 'inline-flex';
            btnAnalyze.style.display = 'inline-flex';
            
            // Retain lastUploadedFile details visually
            if (lastUploadedFile) {
                uploadStatus.textContent = lastUploadedFile.name;
                uploadStatus.style.color = '#60a5fa';
            }
        }
    });

    // --- Piano Roll Zoom Controls ---
    const zoomSlider = document.getElementById('zoom-slider');
    const zoomLabel = document.getElementById('zoom-label');
    if (zoomSlider) {
        const applyZoom = () => {
            const scale = parseFloat(zoomSlider.value);
            if (zoomLabel) zoomLabel.textContent = `${scale.toFixed(1)}x`;
            // Apply scale to the inner SVG via CSS variable on the visualizer
            const vis = document.getElementById('main-visualizer');
            if (vis) {
                const svg = vis.shadowRoot?.querySelector('svg') || vis.querySelector('svg');
                if (svg) {
                    svg.style.transform = `scaleY(${scale})`;
                    svg.style.transformOrigin = 'top left';
                }
            }
        };
        zoomSlider.addEventListener('input', applyZoom);
        // Apply initial zoom after MIDI loads and update bar range bounds
        mainVisualizer.addEventListener('load', () => {
            setTimeout(applyZoom, 200);
            try {
                const ns = mainVisualizer.noteSequence;
                if (ns && ns.totalTime) {
                    const duration = ns.totalTime;
                    let bpm = 120;
                    if (ns.tempos && ns.tempos.length > 0) {
                        bpm = ns.tempos[0].qpm || 120;
                    }
                    const totalBeats = duration * (bpm / 60);
                    const totalBars = Math.ceil(totalBeats / 4);
                    if (typeof updateBarInputsLimit === 'function') {
                        updateBarInputsLimit(totalBars);
                    }
                }
            } catch (err) {
                console.error("Error updating bar range limit:", err);
            }
        });
    }

    // --- Interactive Chat ---
    const chatInput = document.getElementById('chat-input');

    async function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        appendMessage('user', message);
        chatInput.value = '';
        btnSend.disabled = true;

        try {
            const modelName = document.getElementById('cfg-model')?.value || 'gemini-3.1-flash-lite';
            const enableChatRag = document.getElementById('cfg-chat-rag')?.checked || false;
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: message,
                    model_name: modelName,
                    enable_chat_rag: enableChatRag
                })
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }
            const result = await response.json();
            await streamMessage(result.reply);

            // Dynamic graph update from chat
            if (result.new_graph_data && nodes && edges) {
                result.new_graph_data.nodes.forEach(n => {
                    if (!nodes.get(n.id)) {
                        nodes.add(n);
                        rawNodesMap[n.id] = n;
                    }
                });
                result.new_graph_data.edges.forEach(e => {
                    // Check if edge already exists to avoid duplicates
                    const existing = edges.get({
                        filter: item => item.from === e.from && item.to === e.to
                    });
                    if (existing.length === 0) {
                        edges.add(e);
                    }
                });
                // Slightly zoom out or stabilize if needed
                network.stabilize();
            }
        } catch (error) {
            console.error('Chat failed:', error);
            appendMessage('bot', `❌ Chat Error: ${error.message}`);
        } finally {
            btnSend.disabled = false;
        }
    }

    btnSend.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });

    function appendMessage(sender, textOrHtml) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender === 'user' ? 'user-message' : 'bot-message'}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        
        // Escape # inside math blocks before parsing, to prevent KaTeX from breaking
        let processedText = textOrHtml;
        if (sender === 'bot') {
            // Process block math $$...$$
            processedText = processedText.replace(/\$\$([\s\S]*?)\$\$/g, (match, p1) => {
                return '$$' + p1.replace(/#/g, '\\sharp ') + '$$';
            });
            // Process inline math $...$
            processedText = processedText.replace(/\$([^\$]+)\$/g, (match, p1) => {
                return '$' + p1.replace(/#/g, '\\sharp ') + '$';
            });
        }
        bubble.innerHTML = marked.parse(processedText);

        // Render LaTeX in the new bubble
        renderLatex(bubble);

        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    async function streamMessage(text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message bot-message';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        msgDiv.appendChild(bubble);
        chatHistory.appendChild(msgDiv);

        let processedText = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, p1) => {
            return '$$' + p1.replace(/#/g, '\\sharp ') + '$$';
        });
        processedText = processedText.replace(/\$([^\$]+)\$/g, (match, p1) => {
            return '$' + p1.replace(/#/g, '\\sharp ') + '$';
        });

        // Speed depends on length (approx 1.5s total time max)
        const step = Math.max(1, Math.floor(processedText.length / 50));
        let i = 0;
        
        return new Promise(resolve => {
            const interval = setInterval(() => {
                i += step;
                if (i >= processedText.length) {
                    i = processedText.length;
                    bubble.innerHTML = marked.parse(processedText);
                    renderLatex(bubble);
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                    clearInterval(interval);
                    resolve();
                } else {
                    // Show raw markdown while streaming to avoid KaTeX rendering flashes
                    bubble.innerHTML = marked.parse(processedText.substring(0, i)) + '█';
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            }, 30);
        });
    }

    // --- Graph Rendering using vis-network ---
    function renderGraph(graphData) {
        const mergeGraphs = document.getElementById('cfg-merge-graphs')?.checked || false;
        
        if (mergeGraphs && nodes && edges && network) {
            // Dynamic Merge Mode
            graphData.nodes.forEach(n => {
                rawNodesMap[n.id] = n;
                if (!nodes.get(n.id)) {
                    nodes.add({
                        id: n.id,
                        label: n.label,
                        group: n.group,
                        title: "Click to view details"
                    });
                }
            });
            graphData.edges.forEach(e => {
                const existing = edges.get({
                    filter: item => item.from === e.from && item.to === e.to
                });
                if (existing.length === 0) {
                    edges.add({
                        from: e.from,
                        to: e.to,
                        label: e.label,
                        font: { align: 'middle', size: 10, color: '#94a3b8', face: 'Inter', strokeWidth: 0, bold: false },
                        arrows: 'to',
                        color: { color: 'rgba(255,255,255,0.2)' }
                    });
                }
            });
            return;
        }

        // Standard Clear-and-Draw Mode
        // Clear old data
        for (let k in rawNodesMap) delete rawNodesMap[k];
        graphData.nodes.forEach(n => rawNodesMap[n.id] = n);

        nodes = new vis.DataSet(graphData.nodes.map(n => ({
            id: n.id,
            label: n.label,
            group: n.group,
            title: "Click to view details" // Simple tooltip
        })));

        edges = new vis.DataSet(graphData.edges.map(e => ({
            from: e.from,
            to: e.to,
            label: e.label,
            font: { align: 'middle', size: 10, color: '#94a3b8', face: 'Inter', strokeWidth: 0, bold: false },
            arrows: 'to',
            color: { color: 'rgba(255,255,255,0.2)' }
        })));

        const container = document.getElementById('network-container');
        const data = { nodes, edges };
        
        const options = {
            nodes: {
                shape: 'dot',
                size: 20,
                font: { size: 14, color: '#f0f4f8' },
                borderWidth: 2,
                shadow: true
            },
            edges: {
                width: 1,
                smooth: { type: 'continuous' }
            },
            groups: {
                seed: { color: { background: '#ef4444', border: '#b91c1c' } },
                extracted_entity: { color: { background: '#ef4444', border: '#b91c1c' } },
                related: { color: { background: '#3b82f6', border: '#1d4ed8' } },
                instrument: { color: { background: '#10b981', border: '#047857' } },
                theory: { color: { background: '#8b5cf6', border: '#6d28d9' } },
                form: { color: { background: '#f59e0b', border: '#b45309' } }
            },
            physics: {
                forceAtlas2Based: {
                    gravitationalConstant: -50,
                    centralGravity: 0.01,
                    springLength: 100,
                    springConstant: 0.08
                },
                maxVelocity: 50,
                solver: 'forceAtlas2Based',
                timestep: 0.35,
                stabilization: { iterations: 150 }
            },
            interaction: {
                tooltipDelay: 200,
                hover: true
            }
        };

        network = new vis.Network(container, data, options);

        // Details Panel Logic
        const detailsPanel = document.getElementById('node-details-panel');
        const detailsTitle = document.getElementById('details-title');
        const detailsCategory = document.getElementById('details-category');
        const detailsDesc = document.getElementById('details-desc');
        const detailsSource = document.getElementById('details-source');
        
        document.getElementById('close-details-btn').addEventListener('click', () => {
            detailsPanel.classList.add('closed');
        });

        network.on("click", function (params) {
            if (params.nodes.length > 0) {
                const nodeId = params.nodes[0];
                const nodeData = rawNodesMap[nodeId];
                
                detailsTitle.textContent = nodeData.label;
                detailsCategory.textContent = nodeData.group.replace('_', ' ').toUpperCase();
                detailsDesc.textContent = nodeData.title || "No description available.";

                // Smart source URL: if google_search, link to search; else Wikipedia
                const src = nodeData.source || '';
                if (src.startsWith('http')) {
                    detailsSource.href = src;
                    detailsSource.textContent = 'View Source (Google Search)';
                } else {
                    detailsSource.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(nodeData.label)}`;
                    detailsSource.textContent = 'View on Wikipedia';
                }
                
                detailsPanel.classList.remove('closed');
            } else {
                detailsPanel.classList.add('closed');
            }
        });
    }

    // --- History Tab ---
    const historyList = document.getElementById('history-list');
    const btnRefreshHistory = document.getElementById('btn-refresh-history');

    async function loadHistoryList() {
        historyList.innerHTML = '<p class="text-muted" style="padding:1rem;">Loading...</p>';
        try {
            const res = await fetch('/api/analyses');
            const data = await res.json();
            if (!data.analyses || data.analyses.length === 0) {
                historyList.innerHTML = '<p class="text-muted" style="padding:1rem;">No saved analyses yet.</p>';
                return;
            }
            historyList.innerHTML = '';
            data.analyses.forEach(a => {
                const item = document.createElement('div');
                item.className = 'history-item';
                item.innerHTML = `
                    <div style="flex: 1; min-width: 0;">
                        <div class="history-item-title">${a.midi_file}</div>
                        <div class="history-item-meta">${a.timestamp.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5:$6')} · ${a.model}</div>
                    </div>
                    <div class="history-item-actions">
                        <button class="history-action-btn load-btn" data-action="load" title="Load this session to Chat">Load Session</button>
                        <button class="history-action-btn inspect-btn" data-action="inspect" title="Inspect detailed analysis data">Inspect Data</button>
                    </div>
                `;
                
                item.querySelector('.load-btn').addEventListener('click', async (e) => {
                    e.stopPropagation();
                    historyList.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                    await loadAnalysis(a.id, a.midi_file);
                });
                
                item.querySelector('.inspect-btn').addEventListener('click', async (e) => {
                    e.stopPropagation();
                    await inspectAnalysis(a.id, a.midi_file);
                });
                
                item.addEventListener('click', async () => {
                    historyList.querySelectorAll('.history-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                    await loadAnalysis(a.id, a.midi_file);
                });
                
                historyList.appendChild(item);
            });
        } catch(e) {
            historyList.innerHTML = '<p class="text-muted" style="padding:1rem;">Failed to load history.</p>';
        }
    }

    async function inspectAnalysis(id, midiFile) {
        const overlay = document.getElementById('analysis-details-overlay');
        const overlayTitle = document.getElementById('overlay-title');
        const overlaySubtitle = document.getElementById('overlay-subtitle');
        if (!overlay) return;
        
        overlayTitle.textContent = midiFile;
        overlaySubtitle.textContent = 'Loading details...';
        overlay.classList.add('open');
        
        // Show Report tab content by default
        const reportTabBtn = document.querySelector('.overlay-tab-btn[data-tab="tab-report"]');
        if (reportTabBtn) reportTabBtn.click();
        
        try {
            const res = await fetch(`/api/analyses/${encodeURIComponent(id)}/load`);
            if (!res.ok) throw new Error('Failed to load details.');
            const data = await res.json();
            
            overlaySubtitle.textContent = `Timestamp: ${data.timestamp ? data.timestamp.replace(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/, '$1-$2-$3 $4:$5:$6') : 'N/A'} · Model: ${data.model || 'Gemini'}`;
            
            document.getElementById('overlay-report-content').innerHTML = marked.parse(data.final_report || '*No report found.*');
            document.getElementById('overlay-llama-content').textContent = data.llama_analysis || 'No Llama analysis output found.';
            document.getElementById('overlay-m21-content').textContent = JSON.stringify(data.music21 || {}, null, 2);
            document.getElementById('overlay-rag-content').innerHTML = marked.parse(data.rag_context || '*No GraphRAG web context retrieved.*');
            document.getElementById('overlay-cag-content').innerHTML = marked.parse(data.cag_context || '*No CAG local book context retrieved.*');
            
            renderLatex(document.getElementById('overlay-report-content'));
        } catch(e) {
            overlaySubtitle.textContent = 'Failed to load details.';
            document.getElementById('overlay-report-content').innerHTML = `<p style="color: var(--danger); padding:1rem;">Error: ${e.message}</p>`;
        }
    }

    async function loadAnalysis(id, midiFile) {
        appendMessage('bot', `📂 Loading analysis: **${midiFile}**...`);
        // Switch to chat tab
        document.querySelector('[data-target="chat-tab"]').click();
        try {
            const res = await fetch(`/api/analyses/${encodeURIComponent(id)}/load`);
            if (!res.ok) throw new Error('Failed to load analysis.');
            currentSessionId = data.session_id;
            
            // Auto-load the corresponding MIDI file from library silently
            if (data.midi_file) {
                await loadServerMidiSilently(data.midi_file);
            }

            // Render the final report in chat
            const reportBubble = document.createElement('div');
            reportBubble.className = 'message bot-message';
            const bubble = document.createElement('div');
            bubble.className = 'bubble report-bubble';
            bubble.innerHTML = marked.parse(data.final_report || 'No report found.');
            renderLatex(bubble);
            reportBubble.appendChild(bubble);
            chatHistory.appendChild(reportBubble);
            chatHistory.scrollTop = chatHistory.scrollHeight;

            // Render Graph
            if (data.graph_data && data.graph_data.nodes && data.graph_data.nodes.length > 0) {
                renderGraph(data.graph_data);
                if (graphEmptyState) graphEmptyState.style.display = 'none';
                if (networkContainer) networkContainer.style.display = 'block';
            }

            // Update bar inputs limit from historical music21 data
            if (data.music21 && data.music21.total_measures) {
                if (typeof updateBarInputsLimit === 'function') {
                    updateBarInputsLimit(data.music21.total_measures);
                }
            }

            appendMessage('bot', `✅ Analysis loaded. Session active — you can now ask follow-up questions!`);
        } catch(e) {
            appendMessage('bot', `❌ Failed to load: ${e.message}`);
        }
    }

    // Settings slider label
    const tempSlider = document.getElementById('cfg-temperature');
    const tempLabel = document.getElementById('temp-label');
    if (tempSlider) {
        tempSlider.addEventListener('input', () => {
            tempLabel.textContent = parseFloat(tempSlider.value).toFixed(2);
        });
    }

    const showStaffToggle = document.getElementById('cfg-show-staff');
    if (showStaffToggle) {
        // Load preference
        const showStaffPref = localStorage.getItem('showStaffVisualizer') === 'true';
        showStaffToggle.checked = showStaffPref;
        if (showStaffPref) {
            staffVisualizer.classList.add('active');
            staffContainer?.classList.add('active');
        } else {
            staffVisualizer.classList.remove('active');
            staffContainer?.classList.remove('active');
        }

        showStaffToggle.addEventListener('change', (e) => {
            const isChecked = e.target.checked;
            localStorage.setItem('showStaffVisualizer', isChecked);
            if (isChecked) {
                staffVisualizer.classList.add('active');
                staffContainer?.classList.add('active');
            } else {
                staffVisualizer.classList.remove('active');
                staffContainer?.classList.remove('active');
            }
        });
    }

    // --- WaveRoll Custom Settings ---
    const wrNoteColorInput = document.getElementById('cfg-waveroll-note-color');
    const wrBgColorInput = document.getElementById('cfg-waveroll-bg-color');
    const wrFontSizeInput = document.getElementById('cfg-waveroll-font-size');
    const wrFontSizeVal = document.getElementById('waveroll-font-size-val');
    const wrZoomInput = document.getElementById('cfg-waveroll-zoom');
    const wrZoomVal = document.getElementById('waveroll-zoom-val');
    const wrShowTimeInput = document.getElementById('cfg-waveroll-show-time');
    const wrColorByTrackInput = document.getElementById('cfg-waveroll-color-by-track');
    const wrTempoInput = document.getElementById('cfg-waveroll-tempo');
    const wrTempoVal = document.getElementById('waveroll-tempo-val');
    const wrBeatsInput = document.getElementById('cfg-waveroll-beats-per-bar');
    const wrGridSubInput = document.getElementById('cfg-waveroll-grid-sub');
    const wrGridSubVal = document.getElementById('waveroll-grid-sub-val');
    const wrGridSubNumInput = document.getElementById('cfg-waveroll-grid-sub-input');
    const prGridSubNumInput = document.getElementById('pr-grid-sub-input');
    const prGridSubSelect = document.getElementById('pr-grid-sub-select');
    
    // Smooth transition sidebar elements
    const wrTransitionEnabledInput = document.getElementById('cfg-waveroll-transition-enabled');
    const wrTransitionDurationInput = document.getElementById('cfg-waveroll-transition-duration');
    const wrTransitionDurationVal = document.getElementById('waveroll-transition-duration-val');
    const sidebarDurationContainer = document.getElementById('sidebar-duration-container');
    
    const btnGridDouble = document.getElementById('btn-waveroll-grid-double');
    const btnGridHalve = document.getElementById('btn-waveroll-grid-halve');
    const btnPrGridDouble = document.getElementById('btn-pr-grid-double');
    const btnPrGridHalve = document.getElementById('btn-pr-grid-halve');

    const gridSubLabels = {
        '1': '4th (Quarter)',
        '2': '8th (Eighth)',
        '3': 'Triplet 8th (3rd)',
        '4': '16th (Sixteenth)',
        '6': 'Triplet 16th (6th)',
        '8': '32nd (Thirty-second)'
    };
    function updateGridSubLabel(val) {
        if (wrGridSubVal) {
            wrGridSubVal.textContent = gridSubLabels[val] || (val + 'th');
        }
    }

    function updateWaveRollConfig() {
        if (!waveRoll) return;
        const gridSub = wrGridSubNumInput ? parseInt(wrGridSubNumInput.value, 10) : (wrGridSubInput ? parseInt(wrGridSubInput.value, 10) : 4);
        const config = {
            noteColor: wrNoteColorInput?.value || '#ef4444',
            backgroundColor: wrBgColorInput?.value || '#0f172a',
            timeLabelFontSize: parseInt(wrFontSizeInput?.value || '13', 10),
            zoomY: parseFloat(wrZoomInput?.value || '2.5'),
            showTimeGrid: wrShowTimeInput ? wrShowTimeInput.checked : false,
            colorByTrack: wrColorByTrackInput ? wrColorByTrackInput.checked : true,
            tempo: wrTempoInput ? parseInt(wrTempoInput.value, 10) : 120,
            beatsPerBar: wrBeatsInput ? parseInt(wrBeatsInput.value, 10) : 4,
            gridSubdivision: gridSub,
            globalTransitionEnabled: wrTransitionEnabledInput ? wrTransitionEnabledInput.checked : true,
            transitionDuration: wrTransitionDurationInput ? parseFloat(wrTransitionDurationInput.value) : 0.05
        };
        waveRoll.setAttribute('config', JSON.stringify(config));
        
        // Save to localStorage
        localStorage.setItem('waveroll-config', JSON.stringify(config));
    }

    function initWaveRollConfig() {
        try {
            const saved = localStorage.getItem('waveroll-config');
            let config = {
                noteColor: '#ef4444',
                backgroundColor: '#0f172a',
                timeLabelFontSize: 13,
                zoomY: 2.5,
                showTimeGrid: false,
                colorByTrack: true,
                tempo: 120,
                beatsPerBar: 4,
                gridSubdivision: 4,
                globalTransitionEnabled: true,
                transitionDuration: 0.05
            };
            if (saved) {
                config = { ...config, ...JSON.parse(saved) };
            }
            if (wrNoteColorInput) wrNoteColorInput.value = config.noteColor;
            if (wrBgColorInput) wrBgColorInput.value = config.backgroundColor;
            if (wrFontSizeInput) {
                wrFontSizeInput.value = config.timeLabelFontSize;
                if (wrFontSizeVal) wrFontSizeVal.textContent = config.timeLabelFontSize + 'px';
            }
            if (wrZoomInput) {
                wrZoomInput.value = config.zoomY;
                if (wrZoomVal) wrZoomVal.textContent = config.zoomY + 'x';
            }
            if (wrShowTimeInput) {
                wrShowTimeInput.checked = config.showTimeGrid;
            }
            if (wrColorByTrackInput) {
                wrColorByTrackInput.checked = config.colorByTrack;
            }
            if (wrTempoInput) {
                wrTempoInput.value = config.tempo;
                if (wrTempoVal) wrTempoVal.textContent = config.tempo + ' BPM';
            }
            if (wrBeatsInput && config.beatsPerBar !== undefined) {
                wrBeatsInput.value = config.beatsPerBar.toString();
            }
            if (wrGridSubInput && config.gridSubdivision !== undefined) {
                wrGridSubInput.value = config.gridSubdivision.toString();
                updateGridSubLabel(config.gridSubdivision.toString());
            }
            if (wrTransitionEnabledInput) {
                wrTransitionEnabledInput.checked = config.globalTransitionEnabled !== false;
                if (sidebarDurationContainer) {
                    sidebarDurationContainer.style.opacity = wrTransitionEnabledInput.checked ? "1" : "0.4";
                }
            }
            if (wrTransitionDurationInput) {
                wrTransitionDurationInput.value = (config.transitionDuration ?? 0.05).toString();
                if (wrTransitionDurationVal) wrTransitionDurationVal.textContent = (config.transitionDuration ?? 0.05).toFixed(2) + 's';
                wrTransitionDurationInput.disabled = !(config.globalTransitionEnabled !== false);
            }
            if (waveRoll) {
                waveRoll.setAttribute('config', JSON.stringify(config));
            }
        } catch (e) {
            console.error('Failed to init WaveRoll config:', e);
        }
    }

    if (wrNoteColorInput) wrNoteColorInput.addEventListener('input', updateWaveRollConfig);
    if (wrBgColorInput) wrBgColorInput.addEventListener('input', updateWaveRollConfig);
    if (wrFontSizeInput) {
        wrFontSizeInput.addEventListener('input', () => {
            if (wrFontSizeVal) wrFontSizeVal.textContent = wrFontSizeInput.value + 'px';
            updateWaveRollConfig();
        });
    }
    if (wrZoomInput) {
        wrZoomInput.addEventListener('input', () => {
            if (wrZoomVal) wrZoomVal.textContent = wrZoomInput.value + 'x';
            updateWaveRollConfig();
        });
    }
    if (wrShowTimeInput) {
        wrShowTimeInput.addEventListener('change', updateWaveRollConfig);
    }
    if (wrColorByTrackInput) {
        wrColorByTrackInput.addEventListener('change', updateWaveRollConfig);
    }
    if (wrTempoInput) {
        wrTempoInput.addEventListener('input', () => {
            if (wrTempoVal) wrTempoVal.textContent = wrTempoInput.value + ' BPM';
            updateWaveRollConfig();
        });
    }
    if (wrBeatsInput) {
        wrBeatsInput.addEventListener('change', updateWaveRollConfig);
    }
    if (wrTransitionEnabledInput) {
        wrTransitionEnabledInput.addEventListener('change', () => {
            const checked = wrTransitionEnabledInput.checked;
            if (wrTransitionDurationInput) wrTransitionDurationInput.disabled = !checked;
            if (sidebarDurationContainer) sidebarDurationContainer.style.opacity = checked ? "1" : "0.4";
            updateWaveRollConfig();
        });
    }
    if (wrTransitionDurationInput) {
        wrTransitionDurationInput.addEventListener('input', () => {
            const val = parseFloat(wrTransitionDurationInput.value);
            if (wrTransitionDurationVal) wrTransitionDurationVal.textContent = val.toFixed(2) + 's';
            updateWaveRollConfig();
        });
    }
    // Master Synchronization function for Grid Subdivision
    function syncGridSubdivision(val) {
        const intVal = parseInt(val, 10) || 4;
        const strVal = intVal.toString();
        
        // 1. Update number inputs
        if (wrGridSubNumInput) wrGridSubNumInput.value = strVal;
        if (prGridSubNumInput) prGridSubNumInput.value = strVal;
        
        // 2. Update select dropdowns (select option if matches, else show custom/blank)
        if (wrGridSubInput) {
            if (Array.from(wrGridSubInput.options).some(opt => opt.value === strVal)) {
                wrGridSubInput.value = strVal;
            } else {
                wrGridSubInput.value = ""; 
            }
        }
        if (prGridSubSelect) {
            if (Array.from(prGridSubSelect.options).some(opt => opt.value === strVal)) {
                prGridSubSelect.value = strVal;
            } else {
                prGridSubSelect.value = ""; 
            }
        }
        
        // 3. Update the display label
        updateGridSubLabel(strVal);
        
        // 4. Trigger wave-roll update
        updateWaveRollConfig();
    }

    if (wrGridSubInput) {
        wrGridSubInput.addEventListener('change', () => {
            syncGridSubdivision(wrGridSubInput.value);
        });
    }
    if (prGridSubSelect) {
        prGridSubSelect.addEventListener('change', () => {
            syncGridSubdivision(prGridSubSelect.value);
        });
    }
    if (wrGridSubNumInput) {
        wrGridSubNumInput.addEventListener('input', () => {
            const val = parseInt(wrGridSubNumInput.value, 10);
            if (val >= 1 && val <= 64) {
                syncGridSubdivision(val);
            }
        });
    }
    if (prGridSubNumInput) {
        prGridSubNumInput.addEventListener('input', () => {
            const val = parseInt(prGridSubNumInput.value, 10);
            if (val >= 1 && val <= 64) {
                syncGridSubdivision(val);
            }
        });
    }

    function handleDouble() {
        const currentVal = wrGridSubNumInput ? parseInt(wrGridSubNumInput.value, 10) : 4;
        let nextVal = currentVal;
        
        if (currentVal === 1) nextVal = 2;
        else if (currentVal === 2) nextVal = 4;
        else if (currentVal === 3) nextVal = 6;
        else if (currentVal === 4) nextVal = 8;
        else nextVal = Math.min(64, currentVal * 2);
        
        if (nextVal !== currentVal) {
            syncGridSubdivision(nextVal);
        }
    }
    
    function handleHalve() {
        const currentVal = wrGridSubNumInput ? parseInt(wrGridSubNumInput.value, 10) : 4;
        let nextVal = currentVal;
        
        if (currentVal === 8) nextVal = 4;
        else if (currentVal === 6) nextVal = 3;
        else if (currentVal === 4) nextVal = 2;
        else if (currentVal === 2) nextVal = 1;
        else nextVal = Math.max(1, Math.floor(currentVal / 2));
        
        if (nextVal !== currentVal) {
            syncGridSubdivision(nextVal);
        }
    }
    
    if (btnGridDouble) btnGridDouble.addEventListener('click', handleDouble);
    if (btnPrGridDouble) btnPrGridDouble.addEventListener('click', handleDouble);
    if (btnGridHalve) btnGridHalve.addEventListener('click', handleHalve);
    if (btnPrGridHalve) btnPrGridHalve.addEventListener('click', handleHalve);

    initWaveRollConfig();

    btnRefreshHistory?.addEventListener('click', loadHistoryList);
    loadHistoryList(); // auto load on start

    // --- MIDI Library (Server Samples) ---
    async function fetchMidiSamples() {
        if (!midiSampleList) {
            console.warn("MIDI sample list container not found!");
            return;
        }
        midiSampleList.innerHTML = '<div class="loading-spinner">Loading...</div>';
        try {
            console.log("Fetching MIDI samples...");
            const response = await fetch('/api/midi-samples');
            const data = await response.json();
            
            midiSampleList.innerHTML = '';
            if (!data.samples || data.samples.length === 0) {
                midiSampleList.innerHTML = '<div class="sidebar-footer"><p>No samples found</p></div>';
                return;
            }

            data.samples.forEach(filename => {
                const item = document.createElement('div');
                item.className = 'midi-item';
                item.draggable = true;
                item.innerHTML = `
                    <div class="icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 18V5l12-2v13"></path><circle cx="6" cy="18" r="3"></circle><circle cx="18" cy="16" r="3"></circle></svg>
                    </div>
                    <div class="name" title="${filename}">${filename}</div>
                `;
                
                item.onclick = () => loadServerMidi(filename);
                
                item.ondragstart = (e) => {
                    e.dataTransfer.setData('text/plain', filename);
                    e.dataTransfer.effectAllowed = 'copy';
                };
                
                midiSampleList.appendChild(item);
            });
        } catch (error) {
            console.error('Failed to fetch samples:', error);
            midiSampleList.innerHTML = '<div class="sidebar-footer"><p>Error loading library</p></div>';
        }
    }

    async function loadServerMidiSilently(filename) {
        try {
            const response = await fetch(`/api/midi-samples/${encodeURIComponent(filename)}`);
            if (!response.ok) return;
            const blob = await response.blob();
            const file = new File([blob], filename, { type: "audio/midi" });
            
            selectedFile = file;
            lastUploadedFile = file;
            uploadStatus.textContent = file.name;
            uploadStatus.style.color = '#60a5fa';
            
            const objectUrl = URL.createObjectURL(file);
            
            mainVisualizer.setAttribute('src', objectUrl);
            mainPlayer.setAttribute('src', objectUrl);
            staffVisualizer.setAttribute('src', objectUrl);
            
            if (waveRoll) {
                waveRoll.setAttribute('files', JSON.stringify([
                    { "path": objectUrl, "name": file.name, "type": "midi" }
                ]));
            }
            
            playerEmptyState.style.display = 'none';
            playerWrapper.style.display = 'flex';
            
            btnSend.style.display = 'inline-flex';
            btnAnalyze.style.display = 'inline-flex';
        } catch (error) {
            console.error('Error loading midi silently:', error);
        }
    }

    async function loadServerMidi(filename) {
        try {
            appendMessage('bot', `⌛ Loading **${filename}** from library...`);
            const response = await fetch(`/api/midi-samples/${encodeURIComponent(filename)}`);
            const blob = await response.blob();
            const file = new File([blob], filename, { type: "audio/midi" });
            handleMidiFile(file);
        } catch (error) {
            appendMessage('bot', `❌ Error loading file: ${error.message}`);
        }
    }

    // Global drop and dragover prevention to fully guard against default browser downloads
    window.addEventListener("dragover", (e) => {
        e.preventDefault();
    }, false);
    window.addEventListener("drop", (e) => {
        e.preventDefault();
    }, false);

    // Drag and Drop for the whole visualizer area
    if (playerWrapper) {
        playerWrapper.ondragover = (e) => {
            e.preventDefault();
            e.stopPropagation();
            playerWrapper.style.boxShadow = 'inset 0 0 20px var(--accent)';
        };
        playerWrapper.ondragleave = () => {
            playerWrapper.style.boxShadow = 'none';
        };
        playerWrapper.ondrop = (e) => {
            e.preventDefault();
            e.stopPropagation();
            playerWrapper.style.boxShadow = 'none';
            
            // 1. Support dropping local files from local computer
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.mid') || file.name.endsWith('.midi')) {
                    handleMidiFile(file);
                } else {
                    appendMessage('bot', `❌ Please drop a valid MIDI file.`);
                }
                return;
            }
            
            // 2. Support dragging from sidebar sample library
            const filename = e.dataTransfer.getData('text/plain');
            if (filename) {
                loadServerMidi(filename);
            }
        };
    }

    btnRefreshSamples?.addEventListener('click', fetchMidiSamples);
    fetchMidiSamples();

    // --- Measure Inputs Synchronization & Storage ---
    const chatStartBar = document.getElementById('chat-start-bar');
    const chatEndBar = document.getElementById('chat-end-bar');
    const cfgStartBar = document.getElementById('cfg-start-bar');
    const cfgEndBar = document.getElementById('cfg-end-bar');
    
    function updateBarInputsLimit(totalBars) {
        const inputs = [chatStartBar, chatEndBar, cfgStartBar, cfgEndBar];
        inputs.forEach(input => {
            if (input) {
                input.min = '0';
                input.max = totalBars.toString();
            }
        });
        
        // Default start to 0 and end to 0 (entire song / last bar)
        if (chatStartBar) chatStartBar.value = '0';
        if (chatEndBar) chatEndBar.value = '0';
        if (cfgStartBar) cfgStartBar.value = '0';
        if (cfgEndBar) cfgEndBar.value = '0';
        
        syncMeasureInputs('0', '0', 'init');
    }
    
    function syncMeasureInputs(startVal, endVal, source) {
        // Save to localStorage
        localStorage.setItem('cfg_start_measure', startVal || '');
        localStorage.setItem('cfg_end_measure', endVal || '');
        
        // Update chat banner inputs if not triggered by chat banner itself
        if (source !== 'chat') {
            if (chatStartBar) chatStartBar.value = startVal || '';
            if (chatEndBar) chatEndBar.value = endVal || '';
        }
        // Update settings tab inputs if not triggered by settings tab itself
        if (source !== 'settings') {
            if (cfgStartBar) cfgStartBar.value = startVal || '';
            if (cfgEndBar) cfgEndBar.value = endVal || '';
        }
    }
    
    // Bind listeners
    if (chatStartBar) {
        chatStartBar.addEventListener('input', () => {
            syncMeasureInputs(chatStartBar.value, chatEndBar ? chatEndBar.value : '', 'chat');
        });
    }
    if (chatEndBar) {
        chatEndBar.addEventListener('input', () => {
            syncMeasureInputs(chatStartBar ? chatStartBar.value : '', chatEndBar.value, 'chat');
        });
    }
    if (cfgStartBar) {
        cfgStartBar.addEventListener('input', () => {
            syncMeasureInputs(cfgStartBar.value, cfgEndBar ? cfgEndBar.value : '', 'settings');
        });
    }
    if (cfgEndBar) {
        cfgEndBar.addEventListener('input', () => {
            syncMeasureInputs(cfgStartBar ? cfgStartBar.value : '', cfgEndBar.value, 'settings');
        });
    }
    
    // Load stored values on initialize
    const savedStart = localStorage.getItem('cfg_start_measure') || '';
    const savedEnd = localStorage.getItem('cfg_end_measure') || '';
    syncMeasureInputs(savedStart, savedEnd, 'init');

    // --- Graph RAG Language dropdown event listener ---
    const cfgRagLang = document.getElementById('cfg-rag-lang');
    const cfgRagLangCustom = document.getElementById('cfg-rag-lang-custom');
    if (cfgRagLang && cfgRagLangCustom) {
        cfgRagLang.addEventListener('change', () => {
            if (cfgRagLang.value === 'custom') {
                cfgRagLangCustom.style.display = 'block';
            } else {
                cfgRagLangCustom.style.display = 'none';
            }
        });
    }

    // --- History Details Inspect Overlay Drawer ---
    const overlay = document.getElementById('analysis-details-overlay');
    const overlayCloseBtn = document.getElementById('overlay-close-btn');
    const overlayTabBtns = document.querySelectorAll('.overlay-tab-btn');
    const overlayTabContents = document.querySelectorAll('.overlay-tab-content');
    const overlayWidthResizer = document.getElementById('overlay-width-resizer');
    
    if (overlayCloseBtn) {
        overlayCloseBtn.addEventListener('click', () => {
            overlay.classList.remove('open');
        });
    }
    
    // Drawer Tab switching
    overlayTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            overlayTabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const targetTab = btn.getAttribute('data-tab');
            overlayTabContents.forEach(content => {
                if (content.id === targetTab) {
                    content.style.display = 'block';
                } else {
                    content.style.display = 'none';
                }
            });
        });
    });
    
    // Drawer Left edge Col-Resize Dragging width adjustment
    if (overlay && overlayWidthResizer) {
        let isResizing = false;
        
        overlayWidthResizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const newWidth = window.innerWidth - e.clientX;
            if (newWidth > 320 && newWidth < window.innerWidth * 0.9) {
                overlay.style.width = `${newWidth}px`;
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            }
        });
    }

    // --- Dynamic Drag Splitter for Visualizer Height ---
    const visResizer = document.getElementById('vis-resizer');
    const visRollContainer = document.getElementById('roll-container');
    
    if (visResizer && visRollContainer) {
        let isResizing = false;
        let startY = 0;
        let startHeight = 0;
        
        visResizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            startY = e.clientY;
            startHeight = visRollContainer.getBoundingClientRect().height;
            document.body.style.userSelect = 'none';
            document.body.style.cursor = 'row-resize';
        });
        
        document.addEventListener('mousemove', (e) => {
            if (!isResizing) return;
            const deltaY = e.clientY - startY;
            const newHeight = startHeight + deltaY;
            
            // Set height as inline style and disable flex
            visRollContainer.style.flex = 'none';
            visRollContainer.style.height = `${Math.max(150, Math.min(newHeight, 1000))}px`;
            
            // Force Wave-Roll visualizer component to resize and redraw!
            const waveRollEl = document.getElementById('waveroll-visualizer');
            if (waveRollEl && typeof waveRollEl.resize === 'function') {
                waveRollEl.resize();
            }
        });
        
        document.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                document.body.style.userSelect = '';
                document.body.style.cursor = '';
            }
        });
    }

    // Comprehensive and persistent global user gesture audio unlock
    function unlockAudio() {
        try {
            if (typeof Tone !== 'undefined' && Tone.start) {
                Tone.start().catch(() => {});
            }
            if (window.Tone && window.Tone.start) {
                window.Tone.start().catch(() => {});
                if (window.Tone.context && window.Tone.context.rawContext && window.Tone.context.rawContext.resume) {
                    window.Tone.context.rawContext.resume().catch(() => {});
                }
            }
            if (mainPlayer && mainPlayer.player && mainPlayer.player.synth && mainPlayer.player.synth.context) {
                const ctx = mainPlayer.player.synth.context;
                if (ctx && ctx.resume) {
                    ctx.resume().catch(() => {});
                }
            }
        } catch (e) {
            console.error('Audio unlock failed:', e);
        }
    }

    ['click', 'mousedown', 'keydown', 'pointerdown', 'touchstart'].forEach(eventName => {
        document.addEventListener(eventName, unlockAudio, { passive: true });
    });
});
