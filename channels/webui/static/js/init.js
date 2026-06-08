// =============================================================================
// WebSocket Connection Management (Module Level)
// =============================================================================

let wsSocket = null;
let wsReconnectAttempts = 0;
const maxWsReconnectAttempts = 50;

function connectWebSocket() {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const token = window.apiToken || '';
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : '';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws${tokenParam}`;

    try {
        wsSocket = new WebSocket(wsUrl);
        window.socket = wsSocket;  // Keep global reference for send.js
    } catch (e) {
        console.error('Failed to create WebSocket:', e);
        scheduleWsReconnect();
        return;
    }

    wsSocket.onopen = () => {
        console.log('WebSocket connected');
        wsReconnectAttempts = 0;
        isWsConnected = true;
        updateConnectionStatus('connected');
    };

    wsSocket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWebSocketMessage(data);
        } catch (e) {
            console.error('Error parsing WebSocket message:', e);
        }
    };

    wsSocket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        wsSocket = null;
        window.socket = null;
        isWsConnected = false;
        updateConnectionStatus('disconnected');
        scheduleWsReconnect();
    };

    wsSocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Don't close here - onclose will fire after onerror
    };
}

function scheduleWsReconnect() {
    if (wsReconnectAttempts >= maxWsReconnectAttempts) {
        console.error('Max WebSocket reconnection attempts reached');
        return;
    }
    wsReconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(1.5, wsReconnectAttempts - 1), 30000);
    console.log(`WS reconnect attempt ${wsReconnectAttempts} in ${Math.round(delay)}ms`);
    setTimeout(connectWebSocket, delay);
}

function handleWebSocketMessage(data) {
    // Handle typed messages from backend
    if (data.type === 'message_added') {
        handleNewMessage(data.message);
        return;
    }
    if (data.type === 'chat_metadata_updated') {
        if (typeof updateChatTitleBar === 'function') {
            updateChatTitleBar(data.title, data.tags || []);
        }
        loadChats();
        return;
    }
    if (data.type === 'status_updated') {
        if (typeof updateConnectionStatus === 'function') {
            updateConnectionStatus(data.status);
        }
        return;
    }
    // Legacy: handle raw message objects (for backwards compatibility)
    // Add an index if missing to ensure proper handling
    if (data.role && data.content !== undefined) {
        if (data.index === undefined) {
            // Try to determine index from current state
            data.index = lastMessageIndex;
        }
        handleNewMessage(data);
    }
}

function handleNewMessage(msg) {
    // Skip if we're currently streaming - messages will be synced after streaming completes
    if (typeof isStreaming !== 'undefined' && isStreaming) {
        return;
    }
    
    // Only process if we have a valid WebSocket connection
    if (!isWsConnected) return;
    if (!msg || msg.index === undefined) return;
    
    // Validate index is sequential (not older than what we already have)
    if (msg.index < lastMessageIndex) {
        console.log('Skipping old message, index:', msg.index, 'current:', lastMessageIndex);
        return;
    }
    
    // Skip if message already exists (check both exact index and streaming placeholder)
    const existingWrapper = chat.querySelector(`[data-index="${msg.index}"]`);
    if (existingWrapper) {
        console.log('Message already exists at index:', msg.index);
        return;
    }

    renderSingleMessage(msg, msg.index, true);
    // Update lastMessageIndex to be one past the last rendered message
    lastMessageIndex = msg.index + 1;
    scrollToBottom();
    updateTokenUsage();
}

function initSettingsGroupCollapse() {
    const form = document.getElementById('settings-form');
    if (!form) return;

    form.addEventListener('click', (event) => {
        const header = event.target.closest('.settings-group-header');
        if (!header || !form.contains(header)) return;

        const group = header.closest('.settings-group');
        if (!group || group.dataset.moduleSubnavManaged === 'true') return;

        group.classList.toggle('is-open');
    });
}

function initSettingsModuleSubnav() {
    const moduleCategories = new Set(['modules', 'user_modules']);

    const getSection = (category) => document.querySelector(`.settings-section[data-category="${category}"]`);

    const setSubnavActive = (category, groupKey = null) => {
        document.querySelectorAll('.settings-nav-subitem').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.category === category && btn.dataset.group === groupKey);
        });
    };

    const setVisibleSublist = (category = null) => {
        document.querySelectorAll('.settings-nav-sublist').forEach(list => {
            const visible = list.dataset.parentCategory === category;
            list.classList.toggle('visible', visible);
        });
    };

    const showModuleLanding = (category) => {
        if (!moduleCategories.has(category)) return;

        const section = getSection(category);
        if (!section) return;

        const itemsContainer = section.querySelector('.settings-items');
        const groupsGrid = section.querySelector('.settings-groups-grid');

        if (itemsContainer) {
            Array.from(itemsContainer.children).forEach(child => {
                if (!child.classList.contains('settings-groups-grid')) {
                    child.style.display = '';
                }
            });
        }

        if (groupsGrid) {
            groupsGrid.style.display = 'none';
            groupsGrid.querySelectorAll('.settings-group').forEach(group => {
                group.style.display = 'none';
                group.classList.remove('is-open');
                group.dataset.moduleSubnavManaged = 'true';
            });
        }

        setVisibleSublist(category);
        setSubnavActive(category, null);
    };

    const showModuleSettingsGroup = (category, groupKey) => {
        const section = getSection(category);
        if (!section) return;

        if (typeof switchSettingsCategory === 'function') {
            switchSettingsCategory(category);
        }

        const itemsContainer = section.querySelector('.settings-items');
        const groupsGrid = section.querySelector('.settings-groups-grid');
        if (!itemsContainer || !groupsGrid) return;

        Array.from(itemsContainer.children).forEach(child => {
            if (!child.classList.contains('settings-groups-grid')) {
                child.style.display = 'none';
            }
        });

        groupsGrid.style.display = 'block';

        groupsGrid.querySelectorAll('.settings-group').forEach(group => {
            const isTarget = group.dataset.group === groupKey;
            group.style.display = isTarget ? 'block' : 'none';
            group.classList.toggle('is-open', isTarget);
            group.dataset.moduleSubnavManaged = 'true';
        });

        setVisibleSublist(category);
        setSubnavActive(category, groupKey);
    };

    const buildModuleSubnav = (categories) => {
        const nav = document.getElementById('settings-nav');
        if (!nav || !categories) return;

        nav.querySelectorAll('.settings-nav-sublist').forEach(el => el.remove());

        moduleCategories.forEach(category => {
            const categoryData = categories[category];
            const groups = categoryData?.groups;
            const parentBtn = nav.querySelector(`.settings-nav-item[data-category="${category}"]`);
            if (!parentBtn || !groups || typeof groups.entries !== 'function') return;

            const moduleGroups = Array.from(groups.entries())
                .filter(([groupKey]) => groupKey.startsWith(`${category}.settings.`))
                .sort((a, b) => (a[1].title || '').localeCompare(b[1].title || ''));

            if (moduleGroups.length === 0) return;

            const sublist = document.createElement('div');
            sublist.className = 'settings-nav-sublist';
            sublist.dataset.parentCategory = category;

            moduleGroups.forEach(([groupKey, groupData]) => {
                const btn = document.createElement('button');
                btn.type = 'button';
                btn.className = 'settings-nav-subitem';
                btn.dataset.category = category;
                btn.dataset.group = groupKey;
                btn.textContent = groupData.title || groupKey.split('.').pop();
                btn.onclick = (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    showModuleSettingsGroup(category, groupKey);
                };
                sublist.appendChild(btn);
            });

            parentBtn.insertAdjacentElement('afterend', sublist);
        });

        const activeParent = nav.querySelector('.settings-nav-item.active')?.dataset.category || null;
        setVisibleSublist(moduleCategories.has(activeParent) ? activeParent : null);
    };

    if (typeof renderSettingsNav === 'function' && !renderSettingsNav.__moduleSubnavWrapped) {
        const originalRenderSettingsNav = renderSettingsNav;
        renderSettingsNav = function wrappedRenderSettingsNav(categories) {
            originalRenderSettingsNav(categories);
            buildModuleSubnav(categories);
        };
        renderSettingsNav.__moduleSubnavWrapped = true;
    }

    if (typeof renderSettingsForm === 'function' && !renderSettingsForm.__moduleSubnavWrapped) {
        const originalRenderSettingsForm = renderSettingsForm;
        renderSettingsForm = function wrappedRenderSettingsForm(categories) {
            originalRenderSettingsForm(categories);
            moduleCategories.forEach(showModuleLanding);
            const activeCategory = document.querySelector('.settings-nav-item.active')?.dataset.category || null;
            setVisibleSublist(moduleCategories.has(activeCategory) ? activeCategory : null);
        };
        renderSettingsForm.__moduleSubnavWrapped = true;
    }

    if (typeof switchSettingsCategory === 'function' && !switchSettingsCategory.__moduleSubnavWrapped) {
        const originalSwitchSettingsCategory = switchSettingsCategory;
        switchSettingsCategory = function wrappedSwitchSettingsCategory(category) {
            originalSwitchSettingsCategory(category);
            if (moduleCategories.has(category)) {
                showModuleLanding(category);
            } else {
                setVisibleSublist(null);
                setSubnavActive(null, null);
            }
        };
        switchSettingsCategory.__moduleSubnavWrapped = true;
    }
}

// =============================================================================
// Initialization
// =============================================================================

async function init() {
    try {
        requestNotificationPermission();
        document.addEventListener('click', () => {
            if (typeof notificationPermission !== 'undefined' && notificationPermission === 'default') {
                requestNotificationPermission();
            }
        }, { once: true });

        await checkConnection();
        if (isConnected) {
            await restoreCurrentChat();
        }
    } catch (err) {
        console.error('Failed to initialize connection:', err);
        isConnected = false;
        updateConnectionStatus('disconnected');
        scheduleReconnect();
    }

    try {
        const savedFontSize = localStorage.getItem('fontSize');
        if (savedFontSize) {
            document.documentElement.style.setProperty('--font-size-base', `${savedFontSize}px`);
        }

        initSettingsModuleSubnav();
        loadTheme();
        loadChats();
        initTagFilterState();
        initSettingsGroupCollapse();

        window.addEventListener('resize', handleTitleBarResize);

        // ─────────────────────────────────────────────────────────────
        // Safe Sound Default Initialization
        // ─────────────────────────────────────────────────────────────
        Object.entries(SOUND_DEFAULTS).forEach(([id, enabled]) => {
            const key = `${id}Enabled`;
            try {
                if (typeof localStorage !== 'undefined') {
                    const current = localStorage.getItem(key);
                    if (current === null) {
                        localStorage.setItem(key, String(enabled));
                    }
                }
            } catch (e) {
                console.warn('[Init] Storage unavailable, using runtime defaults');
            }
        });

        // ─────────────────────────────────────────────────────────────
        // WebSocket Connection
        // ─────────────────────────────────────────────────────────────
        connectWebSocket();

        // API status polling (this is still needed for API health)
        apiStatusIntervalId = setInterval(() => {
            if (isConnected) {
                checkApiStatus();
            }
        }, CONFIG.API_STATUS_INTERVAL);
    } catch (err) {
        console.error('Failed to initialize UI and polling:', err);
    }
}

init();