// MEXC Trading Bot Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.socket = io();
        this.charts = {};
        this.lastUpdateTime = new Date();
        
        this.init();
    }
    
    init() {
        this.setupSocketListeners();
        this.setupEventListeners();
        this.initializeCharts();
        this.loadInitialData();
    }
    
    setupSocketListeners() {
        // Socket connection events
        this.socket.on('connect', () => {
            console.log('Connected to trading bot dashboard');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from trading bot dashboard');
            this.updateConnectionStatus(false);
        });
        
        // Real-time data updates
        this.socket.on('dashboard_update', (data) => {
            this.updateDashboard(data);
        });
        
        this.socket.on('connected', (data) => {
            console.log('Dashboard connected:', data);
        });
    }
    
    setupEventListeners() {
        // Chart control listeners
        document.getElementById('symbolSelect')?.addEventListener('change', () => {
            this.updatePriceChart();
        });
        
        document.getElementById('timeframeSelect')?.addEventListener('change', () => {
            this.updatePriceChart();
        });
        
        // Manual refresh
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
                e.preventDefault();
                this.refreshData();
            }
        });
    }
    
    initializeCharts() {
        this.createPriceChart();
        this.createPnLChart();
    }
    
    createPriceChart() {
        const ctx = document.getElementById('priceChart');
        if (!ctx) return;
        
        this.charts.priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Price',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cccccc'
                        }
                    },
                    y: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cccccc'
                        }
                    }
                },
                elements: {
                    point: {
                        radius: 0,
                        hoverRadius: 4
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }
    
    createPnLChart() {
        const ctx = document.getElementById('pnlChart');
        if (!ctx) return;
        
        this.charts.pnlChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Daily P&L',
                    data: [],
                    backgroundColor: [],
                    borderColor: [],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cccccc'
                        }
                    },
                    y: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cccccc'
                        }
                    }
                }
            }
        });
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
                statusElement.style.color = '#00ff88';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
                statusElement.style.color = '#ff4757';
            }
        }
    }
    
    updateDashboard(data) {
        this.lastUpdateTime = new Date();
        
        // Update performance metrics
        if (data.performance) {
            this.updatePerformanceMetrics(data.performance);
        }
        
        // Update positions table
        if (data.positions) {
            this.updatePositionsTable(data.positions);
        }
        
        // Update signals table
        if (data.signals) {
            this.updateSignalsTable(data.signals);
        }
        
        // Update timestamp
        this.updateTimestamp();
    }
    
    updatePerformanceMetrics(performance) {
        // Today's P&L
        const todayPnlElement = document.getElementById('todayPnl');
        if (todayPnlElement) {
            todayPnlElement.textContent = `$${performance.today_pnl}`;
            todayPnlElement.className = `metric-value ${performance.today_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        }
        
        // Week P&L
        const weekPnlElement = document.getElementById('weekPnl');
        if (weekPnlElement) {
            weekPnlElement.textContent = `$${performance.week_pnl}`;
            weekPnlElement.className = `metric-value ${performance.week_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        }
        
        // Today's trades
        const todayTradesElement = document.getElementById('todayTrades');
        if (todayTradesElement) {
            todayTradesElement.textContent = performance.today_trades;
        }
        
        // Win rate
        const winRateElement = document.getElementById('winRate');
        if (winRateElement) {
            winRateElement.textContent = `${performance.win_rate_today}%`;
        }
    }
    
    updatePositionsTable(positions) {
        const tableBody = document.getElementById('positionsTableBody');
        if (!tableBody) return;
        
        if (positions.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="7" class="no-data">No active positions</td></tr>';
            return;
        }
        
        tableBody.innerHTML = positions.map(position => `
            <tr>
                <td><strong>${position.symbol}</strong></td>
                <td><span class="position-${position.side.toLowerCase()}">${position.side}</span></td>
                <td>${position.size}</td>
                <td>$${position.entry_price}</td>
                <td>$${position.current_price}</td>
                <td class="${position.pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">
                    $${position.pnl} (${position.pnl_pct}%)
                </td>
                <td>${position.duration}</td>
            </tr>
        `).join('');
    }
    
    updateSignalsTable(signals) {
        const tableBody = document.getElementById('signalsTableBody');
        if (!tableBody) return;
        
        if (signals.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="no-data">No recent signals</td></tr>';
            return;
        }
        
        tableBody.innerHTML = signals.slice(0, 5).map(signal => {
            const time = new Date(signal.timestamp).toLocaleTimeString();
            const confidenceClass = signal.confidence >= 0.8 ? 'confidence-high' : 
                                  signal.confidence >= 0.6 ? 'confidence-medium' : 'confidence-low';
            
            return `
                <tr>
                    <td>${time}</td>
                    <td><strong>${signal.symbol}</strong></td>
                    <td>${signal.strategy}</td>
                    <td><span class="signal-${signal.side.toLowerCase()}">${signal.side}</span></td>
                    <td>$${signal.price}</td>
                    <td class="${confidenceClass}">${(signal.confidence * 100).toFixed(1)}%</td>
                </tr>
            `;
        }).join('');
    }
    
    updateTimestamp() {
        const timestampElement = document.getElementById('updateTime');
        if (timestampElement) {
            timestampElement.textContent = this.lastUpdateTime.toLocaleTimeString();
        }
    }
    
    async updatePriceChart() {
        const symbol = document.getElementById('symbolSelect')?.value || 'BTCUSDT';
        const timeframe = document.getElementById('timeframeSelect')?.value || '1h';
        
        try {
            const response = await fetch(`/api/chart/${symbol}?timeframe=${timeframe}&limit=50`);
            const data = await response.json();
            
            if (data && data.data) {
                const labels = data.data.map(item => {
                    const date = new Date(item.time);
                    return date.toLocaleTimeString();
                });
                
                const prices = data.data.map(item => item.close);
                
                this.charts.priceChart.data.labels = labels;
                this.charts.priceChart.data.datasets[0].data = prices;
                this.charts.priceChart.data.datasets[0].label = `${symbol} Price`;
                this.charts.priceChart.update('none');
            }
        } catch (error) {
            console.error('Error updating price chart:', error);
        }
    }
    
    async loadInitialData() {
        try {
            // Load performance data
            const performanceResponse = await fetch('/api/performance');
            const performanceData = await performanceResponse.json();
            this.updatePerformanceMetrics(performanceData);
            
            // Load positions data
            const positionsResponse = await fetch('/api/positions');
            const positionsData = await positionsResponse.json();
            this.updatePositionsTable(positionsData);
            
            // Load signals data
            const signalsResponse = await fetch('/api/signals');
            const signalsData = await signalsResponse.json();
            this.updateSignalsTable(signalsData);
            
            // Load initial chart
            await this.updatePriceChart();
            
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }
    
    refreshData() {
        console.log('Refreshing dashboard data...');
        this.socket.emit('request_update');
        this.loadInitialData();
    }
    
    // Utility methods
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2
        }).format(value);
    }
    
    formatPercentage(value) {
        return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
    }
    
    getTimeAgo(timestamp) {
        const now = new Date();
        const past = new Date(timestamp);
        const diffInSeconds = Math.floor((now - past) / 1000);
        
        if (diffInSeconds < 60) return `${diffInSeconds}s ago`;
        if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
        if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
        return `${Math.floor(diffInSeconds / 86400)}d ago`;
    }
}

// Auto-refresh functionality
class AutoRefresh {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.interval = null;
        this.refreshRate = 30000; // 30 seconds
        
        this.start();
    }
    
    start() {
        this.interval = setInterval(() => {
            this.dashboard.updatePriceChart();
        }, this.refreshRate);
    }
    
    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }
    
    setRefreshRate(rate) {
        this.refreshRate = rate;
        this.stop();
        this.start();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing MEXC Trading Bot Dashboard...');
    
    const dashboard = new TradingDashboard();
    const autoRefresh = new AutoRefresh(dashboard);
    
    // Global error handling
    window.addEventListener('error', (error) => {
        console.error('Dashboard error:', error);
    });
    
    // Visibility change handling (pause updates when tab is not visible)
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            autoRefresh.stop();
        } else {
            autoRefresh.start();
            dashboard.refreshData();
        }
    });
    
    console.log('Dashboard initialized successfully');
}); 