import { test, expect } from "@playwright/test";

test("shopping flow: cards, explicit inventory, comparison, exclusions, restore and mobile", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("5 件商品");
  await page.getByRole("button", { name: /500 元以内/ }).click();
  await expect(page.locator(".card")).toHaveCount(2);
  await page.locator(".card input").nth(0).check();
  await page.locator(".card input").nth(1).check();
  await page.locator("#compare").click();
  await expect(page.locator(".comparison")).toHaveCount(1);
  await page
    .locator('[data-product-id="shoe-002"]')
    .getByRole("button", { name: "查库存" })
    .click();
  await expect(page.locator(".card")).toHaveCount(1);
  await expect(page.locator(".card")).toContainText("shoe-002");
  await page.locator("#query").fill("推荐运动鞋");
  await page.locator("#send").click();
  await expect(page.locator(".card")).toHaveCount(2);
  await page
    .locator('[data-product-id="shoe-001"]')
    .getByRole("button", { name: "排除" })
    .click();
  await expect(page.locator(".card")).toHaveCount(1);
  await expect(page.locator(".side-note")).toContainText("shoe-001");
  const count = await page.locator(".message").count();
  await page.reload();
  await expect(page.locator(".message")).toHaveCount(count);
  await expect(page.locator("#budget")).toHaveValue("500");
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({ path: "test-results/mobile.png", fullPage: true });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.screenshot({ path: "test-results/desktop.png", fullPage: true });
  await page.locator("#new-chat").click();
  await expect(page.locator(".message")).toHaveCount(0);
  await expect(page.locator("#budget")).toHaveValue("");
  expect(errors).toEqual([]);
});

test("image upload is honest, follows up, validates failures, safely renders input", async ({
  page,
}) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("5 件商品");
  await page
    .locator("#image")
    .setInputFiles({
      name: "reference.png",
      mimeType: "image/png",
      buffer: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAAFUlEQVR4nGP8//8/AzbAhFV00EoAAFbUAw037MyjAAAAAElFTkSuQmCC",
        "base64",
      ),
    });
  await page.locator("#query").fill("白色运动鞋");
  await page.locator("#send").click();
  await expect(page.locator(".message.assistant")).toContainText(
    "仅按文字检索",
  );
  await expect(page.locator(".card").first()).toBeVisible();
  await page.locator("#query").fill("有货吗");
  await page.locator("#send").click();
  await expect(page.locator(".message.assistant")).toHaveCount(2);
  await page.locator("#query").fill("<img src=x onerror=alert(1)>");
  await page.locator("#send").click();
  await expect(page.locator(".message.assistant")).toHaveCount(3);
  await expect(page.locator(".message.user img")).toHaveCount(0);
  await page
    .locator("#image")
    .setInputFiles({
      name: "bad.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("bad"),
    });
  await expect(page.getByRole("alert")).toContainText("请选择");
});

test("late response cannot overwrite a new conversation", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("status")).toContainText("5 件商品");
  let release!: () => void;
  const wait = new Promise<void>((r) => (release = r));
  await page.route("**/chat/stream", async (route) => {
    await wait;
    await route
      .fulfill({
        contentType: "text/event-stream",
        body: 'event: result\ndata: {"answer":"STALE RESULT"}\n\n',
      })
      .catch(() => {});
  });
  await page.locator("#query").fill("运动鞋");
  await page.locator("#send").click();
  await expect(page.getByRole("button", { name: "停止等待" })).toBeVisible();
  await page.locator("#new-chat").click();
  release();
  await expect(page.locator(".message")).toHaveCount(0);
  await expect(page.locator("#welcome")).toBeVisible();
});

test("401 and service errors preserve input and show actionable recovery", async ({
  page,
}) => {
  await page.route("**/ui/config", (r) =>
    r.fulfill({ status: 401, json: { detail: "token required" } }),
  );
  await page.goto("/");
  await expect(page.locator("dialog[open]")).toBeVisible();
  await page.getByRole("button", { name: "关闭连接设置" }).click();
  await page.route("**/chat/stream", (r) =>
    r.fulfill({ status: 503, json: { detail: "offline" } }),
  );
  await page.locator("#query").fill("运动鞋");
  await page.locator("#send").click();
  await expect(page.locator("#error")).toContainText("暂时不可用");
  await expect(page.locator("#query")).toHaveValue("运动鞋");
});
