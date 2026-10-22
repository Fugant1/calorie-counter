import React, { useState } from "react";

export type AppLanguage = "pt" | "en";
export type AppView = "home" | "chat" | "plan" | "settings";

export interface ProbableDailyFood {
  food_id: string;
  meal_name: string;
  display_name: string;
  estimated_grams: number;
  macros: {
    calories_kcal: number;
    protein_g: number;
    carbs_g: number;
    fat_g: number;
  };
}

export interface DailyCalorieExchangeRow {
  id: string;
  meal_name: string;
  primary_food_name: string;
  primary_grams: number;
  primary_calories_kcal: number;
  replacement_food_name: string;
  replacement_grams: number;
  replacement_calories_kcal: number;
  calorie_delta_kcal: number;
  applied_today: boolean;
}

export interface FoodEatenRow {
  id: string;
  meal_name: string;
  food_name: string;
  grams: number;
  calories_kcal: number;
  protein_g: number;
  carbs_g: number;
  fat_g: number;
  exchanged_from_food?: string | null;
}

export interface ChatMessageRow {
  id: string;
  conversation_id: string;
  user_id: string;
  sender_role: "user" | "assistant" | "system";
  content: string;
  detected_intent?: string | null;
  registered_meal_id?: string | null;
  node_decision_path: string[];
  tokens_used: number;
  created_at_iso: string;
}

export interface UserDashboardResponse {
  user: {
    id: string;
    cpf: string;
    phone: string;
    address: string;
    email: string;
    password_hash: string;
    full_name?: string | null;
  };
  active_plan?: {
    title: string;
    raw_extracted_text: string;
  } | null;
  daily_totals: {
    total_calories_kcal: number;
    target_calories_kcal: number;
    remaining_calories_kcal: number;
    calorie_exchange_daily_delta_kcal: number;
    total_protein_g: number;
    total_carbs_g: number;
    total_fat_g: number;
  };
  probable_daily_foods: ProbableDailyFood[];
  daily_calorie_exchanges: DailyCalorieExchangeRow[];
  foods_eaten: FoodEatenRow[];
  chat_messages: ChatMessageRow[];
}

const TRANSLATIONS: Record<AppLanguage, Record<string, string>> = {
  pt: {
    navHome: "🏠 Início",
    navChat: "💬 Chat IA",
    navPlan: "📋 Plano Alimentar",
    navSettings: "⚙️ Configurações",
    entranceBadge: "PERFORMANCE NUTRITION APP",
    entranceTitle: "Bem-vindo ao Apex Fit AI",
    entranceDesc:
      "Informe seu login e senha para acessar o painel esportivo, conversar com o Coach IA ou ver seu Plano Alimentar.",
    signInTab: "Entrar (Login)",
    signUpTab: "Criar Conta",
    loginLabel: "E-mail ou CPF",
    passwordLabel: "Senha",
    enterAppBtn: "🚀 Entrar no App",
    createAccountBtn: "🔥 Criar Conta de Atleta",
    athleteHubTag: "PAINEL DO ATLETA",
    helloAthlete: "Olá",
    kcalEaten: "Calorias Consumidas (kcal)",
    kcalTarget: "Meta Diária (kcal)",
    kcalRemaining: "Calorias Restantes (kcal)",
    kcalDelta: "Saldo de Substituições (kcal)",
    chatPortalTitle: "Chat Nutricional & Registro de Refeições",
    chatPortalDesc:
      'Registre o que comeu em linguagem natural (ex: "Comi 3 ovos cozidos com 2 pães de forma") ou tire dúvidas com seu Coach IA. Corrige typos ("ovio", "ovoos") via Small LLM JSON.',
    chatPortalCta: "Abrir Chat Nutricional →",
    planPortalTitle: "Meu Plano Alimentar (.PDF / .TXT)",
    planMissingBadge: "NENHUM PLANO — REGISTRE AGORA",
    planActiveBadge: "PLANO ATIVO REGISTRADO",
    planMissingDesc:
      "Nenhum plano alimentar registrado ainda! Clique aqui para enviar seu PDF ou TXT e registrar seu plano.",
    planMissingCta: "Registrar Plano Alimentar Agora →",
    planActiveCta: "Ver Meu Plano Alimentar →",
    registerPlanHeading: "Registre seu Plano Alimentar (.PDF ou .TXT)",
    updatePlanHeading: "Atualizar / Substituir Plano Alimentar (.PDF ou .TXT)",
    loadSamplePdfBtn: '📄 Carregar "Plano Alimentar Teste.pdf"',
    savePlanBtn: "🔥 Salvar Plano Alimentar",
    probableFoodsHeading: "🥗 Alimentos do Plano Para Hoje",
    exchangesHeading: "🔄 Lista de Substituições Calóricas",
    eatTodayBtn: "+ Comer Hoje",
    swapBtn: "Trocar & Registrar",
    settingsLangHeading: "🌐 Idioma do Aplicativo",
    settingsLangDesc: "Escolha o idioma do aplicativo (Português ou Inglês):",
    logoutBtn: "🚪 Sair da Conta (Logout)",
  },
  en: {
    navHome: "🏠 Home",
    navChat: "💬 AI Chat",
    navPlan: "📋 Nutrition Plan",
    navSettings: "⚙️ Settings",
    entranceBadge: "PERFORMANCE NUTRITION APP",
    entranceTitle: "Welcome to Apex Fit AI",
    entranceDesc:
      "Enter your login and password to access your athlete home, chat with your AI Coach, or view your registered Nutrition Plan.",
    signInTab: "Sign In (Login)",
    signUpTab: "Create Account",
    loginLabel: "Email or CPF",
    passwordLabel: "Password",
    enterAppBtn: "🚀 Enter App",
    createAccountBtn: "🔥 Create Athlete Account",
    athleteHubTag: "ATHLETE PERFORMANCE HUB",
    helloAthlete: "Hello",
    kcalEaten: "Calories Eaten (kcal)",
    kcalTarget: "Daily Target (kcal)",
    kcalRemaining: "Remaining Calories (kcal)",
    kcalDelta: "Exchange Delta (kcal)",
    chatPortalTitle: "AI Nutrition & Meal Logger Chat",
    chatPortalDesc:
      'Log meals in natural language (e.g. "I ate 3 boiled eggs with 2 slices of bread") or ask coaching questions. Fixes typos ("ovio", "ovoos") via Small LLM JSON.',
    chatPortalCta: "Open AI Nutrition Chat →",
    planPortalTitle: "My Nutrition Plan (.PDF / .TXT)",
    planMissingBadge: "NO PLAN — REGISTER NOW",
    planActiveBadge: "ACTIVE PLAN REGISTERED",
    planMissingDesc:
      "No nutrition plan registered yet! Click here to upload your nutritionist PDF or TXT file and register your plan.",
    planMissingCta: "Register Nutrition Plan Now →",
    planActiveCta: "View Registered Nutrition Plan →",
    registerPlanHeading: "Register Your Nutrition Plan (.PDF or .TXT)",
    updatePlanHeading: "Update / Replace Nutrition Plan (.PDF or .TXT)",
    loadSamplePdfBtn: '📄 Load "Plano Alimentar Teste.pdf"',
    savePlanBtn: "🔥 Save Nutrition Plan",
    probableFoodsHeading: "🥗 Probable Daily Foods (Active Plan)",
    exchangesHeading: "🔄 Daily Calorie Exchanges",
    eatTodayBtn: "+ Eat Today",
    swapBtn: "Swap & Log",
    settingsLangHeading: "🌐 App Language / Idioma",
    settingsLangDesc: "Switch the application language between Portuguese and English:",
    logoutBtn: "🚪 Sign Out (Logout)",
  },
};

function arrayBufferToHex(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const hex: string[] = [];
  for (let i = 0; i < bytes.length; i++) {
    hex.push(bytes[i].toString(16).padStart(2, "0"));
  }
  return hex.join("");
}

export function App() {
  const [lang, setLang] = useState<AppLanguage>("pt");
  const [authMode, setAuthMode] = useState<"login" | "signup">("login");
  const [activeView, setActiveView] = useState<AppView>("home");
  const [authError, setAuthError] = useState("");

  // Login & Registration fields
  const [loginInput, setLoginInput] = useState("gabriel.silva@calorie.ai");
  const [password, setPassword] = useState("NutriPlan#2026");
  const [fullName, setFullName] = useState("Gabriel Silva");
  const [cpf, setCpf] = useState("123.456.789-01");
  const [phone, setPhone] = useState("+55 11 99876-5432");
  const [address, setAddress] = useState("Av. Paulista, 1000 - São Paulo, SP");
  const [email, setEmail] = useState("gabriel.silva@calorie.ai");

  // Plan & Chat states
  const [filename, setFilename] = useState("Plano Alimentar Teste.pdf");
  const [fileBytesHex, setFileBytesHex] = useState<string | null>(null);
  const [planText, setPlanText] = useState("");
  const [chatInput, setChatInput] = useState("");
  const [dashboard, setDashboard] = useState<UserDashboardResponse | null>(null);

  const t = (key: string) => TRANSLATIONS[lang][key] ?? key;

  async function handleLogin() {
    setAuthError("");
    const res = await fetch("/api/v1/users/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ login: loginInput, password }),
    });
    const data = await res.json();
    if (data.error) {
      setAuthError(data.error);
      return;
    }
    setDashboard(data);
    setActiveView("home");
  }

  async function handleRegisterUser() {
    setAuthError("");
    const res = await fetch("/api/v1/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ full_name: fullName, cpf, phone, address, email, password }),
    });
    const data = await res.json();
    if (data.error) {
      setAuthError(data.error);
      return;
    }
    setDashboard(data);
    setActiveView("home");
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setFilename(file.name);
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        setFileBytesHex(arrayBufferToHex(reader.result));
      }
    };
    reader.readAsArrayBuffer(file);
  }

  async function handleLoadSamplePdf() {
    if (!dashboard) return;
    const res = await fetch("/api/v1/plans/load-sample-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: dashboard.user.id }),
    });
    const data = await res.json();
    if (!data.error) {
      setDashboard(data);
      if (data.active_plan?.raw_extracted_text) {
        setPlanText(data.active_plan.raw_extracted_text);
      }
    }
  }

  async function handleUploadPlan() {
    if (!dashboard) return;
    const res = await fetch("/api/v1/plans/upload", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: dashboard.user.id,
        filename,
        file_bytes_hex: fileBytesHex,
        content_text: planText,
        as_pdf_binary: filename.toLowerCase().endsWith(".pdf"),
      }),
    });
    const data = await res.json();
    if (!data.error) {
      setDashboard(data);
      if (data.active_plan?.raw_extracted_text) {
        setPlanText(data.active_plan.raw_extracted_text);
      }
    }
  }

  async function handleSendChatMessage() {
    if (!dashboard || !chatInput.trim()) return;
    const msg = chatInput.trim();
    setChatInput("");
    const res = await fetch("/api/v1/chat/send", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: dashboard.user.id, message: msg }),
    });
    const data = await res.json();
    if (!data.error) {
      setDashboard(data);
    }
  }

  async function handleLogFood(mealName: string, foodName: string, grams: number) {
    if (!dashboard) return;
    const res = await fetch("/api/v1/foods/log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: dashboard.user.id,
        meal_name: mealName,
        food_name: foodName,
        grams,
      }),
    });
    const data = await res.json();
    if (!data.error) {
      setDashboard(data);
    }
  }

  async function handleApplyExchange(exchangeId: string) {
    if (!dashboard) return;
    const res = await fetch("/api/v1/exchanges/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: dashboard.user.id, exchange_id: exchangeId }),
    });
    const data = await res.json();
    if (!data.error) {
      setDashboard(data);
    }
  }

  const redButtonStyle: React.CSSProperties = {
    background: "linear-gradient(135deg, #dc2626 0%, #b91c1c 100%)",
    color: "#ffffff",
    fontWeight: 800,
    border: "none",
    padding: "12px 20px",
    borderRadius: 10,
    cursor: "pointer",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  };

  const outlineButtonStyle: React.CSSProperties = {
    background: "#ffffff",
    color: "#dc2626",
    border: "2px solid #dc2626",
    fontWeight: 800,
    padding: "10px 18px",
    borderRadius: 10,
    cursor: "pointer",
    textTransform: "uppercase",
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f8fafc", color: "#0f172a", fontFamily: "system-ui, sans-serif" }}>
      {/* TOP SPORTS BAR */}
      <header
        style={{
          background: "linear-gradient(135deg, #991b1b 0%, #dc2626 55%, #ef4444 100%)",
          color: "#ffffff",
          padding: "14px 28px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ fontWeight: 900, fontSize: 20, fontStyle: "italic", letterSpacing: 0.8 }}>
          ⚡ APEX FIT AI
        </div>
        {dashboard && (
          <nav style={{ display: "flex", gap: 8 }}>
            <button style={outlineButtonStyle} onClick={() => setActiveView("home")}>
              {t("navHome")}
            </button>
            <button style={outlineButtonStyle} onClick={() => setActiveView("chat")}>
              {t("navChat")}
            </button>
            <button style={outlineButtonStyle} onClick={() => setActiveView("plan")}>
              {t("navPlan")}
            </button>
            <button style={outlineButtonStyle} onClick={() => setActiveView("settings")}>
              {t("navSettings")}
            </button>
          </nav>
        )}
      </header>

      {/* ENTRANCE VIEW (LOGIN / SIGN UP) */}
      {!dashboard ? (
        <main style={{ maxWidth: 480, margin: "48px auto", padding: 32, background: "#ffffff", borderRadius: 20, borderTop: "6px solid #dc2626", boxShadow: "0 20px 45px rgba(220,38,38,0.14)" }}>
          <span style={{ background: "#fee2e2", color: "#dc2626", fontWeight: 800, fontSize: 11, padding: "4px 12px", borderRadius: 999 }}>
            {t("entranceBadge")}
          </span>
          <h1 style={{ marginTop: 12, marginBottom: 6 }}>{t("entranceTitle")}</h1>
          <p style={{ color: "#64748b", fontSize: 14 }}>{t("entranceDesc")}</p>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, margin: "16px 0" }}>
            <button style={authMode === "login" ? redButtonStyle : outlineButtonStyle} onClick={() => setAuthMode("login")}>
              {t("signInTab")}
            </button>
            <button style={authMode === "signup" ? redButtonStyle : outlineButtonStyle} onClick={() => setAuthMode("signup")}>
              {t("signUpTab")}
            </button>
          </div>

          {authMode === "login" ? (
            <div style={{ display: "grid", gap: 12 }}>
              <label>{t("loginLabel")}</label>
              <input value={loginInput} onChange={(e) => setLoginInput(e.target.value)} />
              <label>{t("passwordLabel")}</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} />
              <button style={redButtonStyle} onClick={handleLogin}>
                {t("enterAppBtn")}
              </button>
            </div>
          ) : (
            <div style={{ display: "grid", gap: 10 }}>
              <input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Full Name" />
              <input value={cpf} onChange={(e) => setCpf(e.target.value)} placeholder="CPF" />
              <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" />
              <input value={phone} onChange={(e) => setPhone(e.target.value)} placeholder="Phone" />
              <input value={address} onChange={(e) => setAddress(e.target.value)} placeholder="Address" />
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="Password" />
              <button style={redButtonStyle} onClick={handleRegisterUser}>
                {t("createAccountBtn")}
              </button>
            </div>
          )}

          {authError && <p style={{ color: "#dc2626", fontWeight: 700, marginTop: 12 }}>{authError}</p>}
        </main>
      ) : (
        <main style={{ maxWidth: 1180, margin: "0 auto", padding: 28 }}>
          {/* HOME VIEW */}
          {activeView === "home" && (
            <section>
              <h2>
                {t("helloAthlete")}, {dashboard.user.full_name ?? dashboard.user.email}!
              </h2>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
                <div style={{ background: "#fff", borderTop: "4px solid #dc2626", padding: 18, borderRadius: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: 900, color: "#dc2626" }}>{dashboard.daily_totals.total_calories_kcal.toFixed(1)}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{t("kcalEaten")}</div>
                </div>
                <div style={{ background: "#fff", borderTop: "4px solid #dc2626", padding: 18, borderRadius: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: 900, color: "#dc2626" }}>{dashboard.daily_totals.target_calories_kcal.toFixed(0)}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{t("kcalTarget")}</div>
                </div>
                <div style={{ background: "#fff", borderTop: "4px solid #dc2626", padding: 18, borderRadius: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: 900, color: "#dc2626" }}>{dashboard.daily_totals.remaining_calories_kcal.toFixed(1)}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{t("kcalRemaining")}</div>
                </div>
                <div style={{ background: "#fff", borderTop: "4px solid #dc2626", padding: 18, borderRadius: 14, textAlign: "center" }}>
                  <div style={{ fontSize: 28, fontWeight: 900, color: "#dc2626" }}>{dashboard.daily_totals.calorie_exchange_daily_delta_kcal.toFixed(1)}</div>
                  <div style={{ fontSize: 12, color: "#64748b" }}>{t("kcalDelta")}</div>
                </div>
              </div>

              {/* TWO PRIMARY HOME PORTALS: 1) AI CHAT  2) NUTRITION PLAN */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 22 }}>
                <div
                  onClick={() => setActiveView("chat")}
                  style={{ background: "#fff", border: "2px solid #fca5a5", borderRadius: 18, padding: 26, cursor: "pointer" }}
                >
                  <h3>💬 {t("chatPortalTitle")}</h3>
                  <p style={{ color: "#64748b", fontSize: 14 }}>{t("chatPortalDesc")}</p>
                  <strong style={{ color: "#dc2626" }}>{t("chatPortalCta")}</strong>
                </div>

                <div
                  onClick={() => setActiveView("plan")}
                  style={{ background: "#fff", border: "2px solid #fca5a5", borderRadius: 18, padding: 26, cursor: "pointer" }}
                >
                  <span style={{ background: "#fee2e2", color: "#dc2626", fontWeight: 800, fontSize: 11, padding: "4px 10px", borderRadius: 999 }}>
                    {dashboard.active_plan ? t("planActiveBadge") : t("planMissingBadge")}
                  </span>
                  <h3 style={{ marginTop: 10 }}>📋 {t("planPortalTitle")}</h3>
                  <p style={{ color: "#64748b", fontSize: 14 }}>
                    {dashboard.active_plan ? dashboard.active_plan.title : t("planMissingDesc")}
                  </p>
                  <strong style={{ color: "#dc2626" }}>
                    {dashboard.active_plan ? t("planActiveCta") : t("planMissingCta")}
                  </strong>
                </div>
              </div>
            </section>
          )}

          {/* CHAT VIEW */}
          {activeView === "chat" && (
            <section style={{ background: "#fff", padding: 24, borderRadius: 16, borderLeft: "6px solid #dc2626" }}>
              <h2>💬 {t("chatPortalTitle")}</h2>
              <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 16, minHeight: 260, marginBottom: 12 }}>
                {dashboard.chat_messages.map((m) => (
                  <div key={m.id} style={{ marginBottom: 12 }}>
                    <strong style={{ color: m.sender_role === "user" ? "#dc2626" : "#0f172a" }}>
                      {m.sender_role.toUpperCase()}:
                    </strong>{" "}
                    {m.content}
                  </div>
                ))}
              </div>
              <div style={{ display: "flex", gap: 10 }}>
                <input
                  style={{ flex: 1 }}
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder='Ex: "Comi 3 ovos cozidos com 2 pães de forma"'
                />
                <button style={redButtonStyle} onClick={handleSendChatMessage}>
                  Send
                </button>
              </div>
            </section>
          )}

          {/* NUTRITION PLAN VIEW (REGISTER IF MISSING, OR VIEW DETAILS) */}
          {activeView === "plan" && (
            <section>
              <div style={{ background: "#fff", padding: 24, borderRadius: 16, borderLeft: "6px solid #dc2626", marginBottom: 22 }}>
                <h2>{dashboard.active_plan ? t("updatePlanHeading") : t("registerPlanHeading")}</h2>
                <div style={{ display: "flex", gap: 12, marginBottom: 12 }}>
                  <input type="file" accept=".pdf,.txt" onChange={handleFileChange} />
                  <button style={outlineButtonStyle} onClick={handleLoadSamplePdf}>
                    {t("loadSamplePdfBtn")}
                  </button>
                  <button style={redButtonStyle} onClick={handleUploadPlan}>
                    {t("savePlanBtn")}
                  </button>
                </div>
                <textarea
                  rows={5}
                  style={{ width: "100%" }}
                  value={planText}
                  onChange={(e) => setPlanText(e.target.value)}
                />
              </div>

              {dashboard.active_plan && (
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  <div style={{ background: "#fff", padding: 20, borderRadius: 16 }}>
                    <h3>{t("probableFoodsHeading")}</h3>
                    {dashboard.probable_daily_foods.map((f) => (
                      <div key={f.food_id} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #e2e8f0" }}>
                        <span>
                          <strong>{f.display_name}</strong> ({f.estimated_grams}g — {f.macros.calories_kcal} kcal)
                        </span>
                        <button style={redButtonStyle} onClick={() => handleLogFood(f.meal_name, f.display_name, f.estimated_grams)}>
                          {t("eatTodayBtn")}
                        </button>
                      </div>
                    ))}
                  </div>
                  <div style={{ background: "#fff", padding: 20, borderRadius: 16 }}>
                    <h3>{t("exchangesHeading")}</h3>
                    {dashboard.daily_calorie_exchanges.map((ex) => (
                      <div key={ex.id} style={{ display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #e2e8f0" }}>
                        <span>
                          {ex.primary_food_name} → <strong>{ex.replacement_food_name}</strong> ({ex.calorie_delta_kcal} kcal)
                        </span>
                        <button style={outlineButtonStyle} onClick={() => handleApplyExchange(ex.id)}>
                          {t("swapBtn")}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* SETTINGS VIEW (LANGUAGE SWITCHER PT / EN) */}
          {activeView === "settings" && (
            <section style={{ background: "#fff", padding: 24, borderRadius: 16, borderLeft: "6px solid #dc2626" }}>
              <h2>{t("settingsLangHeading")}</h2>
              <p style={{ color: "#64748b" }}>{t("settingsLangDesc")}</p>
              <div style={{ display: "flex", gap: 12, marginBottom: 24 }}>
                <button style={lang === "pt" ? redButtonStyle : outlineButtonStyle} onClick={() => setLang("pt")}>
                  🇧🇷 Português (Brasil)
                </button>
                <button style={lang === "en" ? redButtonStyle : outlineButtonStyle} onClick={() => setLang("en")}>
                  🇺🇸 English (US)
                </button>
              </div>
              <button style={outlineButtonStyle} onClick={() => setDashboard(null)}>
                {t("logoutBtn")}
              </button>
            </section>
          )}
        </main>
      )}
    </div>
  );
}
