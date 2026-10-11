export type Card = {
  id: string;
  title: string;
  category: string;
  price: number;
  rating: number;
  stock: number | null;
  inventory_source: string;
  reasons: string[];
};
export type Constraints = {
  max_price: number | null;
  category: string | null;
  excluded_product_ids: string[];
};
export type Reply = {
  answer: string;
  intent: string;
  cards: Card[];
  constraints: Constraints;
  tool_trace: { tool: string; ok: boolean }[];
  tool_calls: number;
  context: { fallback_used?: boolean };
  session_id: string;
};
export type Message = {
  role: "user" | "assistant";
  text: string;
  reply?: Reply;
};
export type Config = {
  categories: string[];
  product_count: number;
  image_search_available: boolean;
  inventory_source: string;
  reasoner_backend: string;
};
export type Intent = "auto" | "search" | "inventory" | "compare";
