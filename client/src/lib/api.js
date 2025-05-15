/**
 * API utility functions for making requests to the backend
 */

export async function apiRequest(method, url, data) {
  const res = await fetch(url, {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
    body: data ? JSON.stringify(data) : undefined,
    credentials: "include",
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`${res.status}: ${errorText || res.statusText}`);
  }

  return res;
}

export async function uploadFile(url, file, additionalData) {
  const formData = new FormData();
  formData.append("file", file);

  if (additionalData) {
    Object.entries(additionalData).forEach(([key, value]) => {
      formData.append(key, value);
    });
  }

  const res = await fetch(url, {
    method: "POST",
    body: formData,
    credentials: "include",
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`${res.status}: ${errorText || res.statusText}`);
  }

  return res;
}
