# frozen_string_literal: true
# Pure render logic for the wiki generator — depends ONLY on kramdown (no Jekyll),
# so it can be unit-tested with plain ruby AND required by the Jekyll plugin.
#
# Pipeline (order matters):
#   1. protect fenced/inline CODE      -> placeholders (restored BEFORE kramdown)
#   2. protect display+inline MATH     -> placeholders (restored AFTER kramdown)
#      (removing $...$ before kramdown avoids: GFM table misparse from | in math,
#       underscore->emphasis, and backslash mangling)
#   3. rewrite internal .md links      (now operates on prose only)
#   4. restore CODE                    (back to ``` fences so kramdown highlights them)
#   5. kramdown render
#   6. restore MATH as \(..\) / \[..\] (post-render: backslashes survive; MathJax reads them)
require 'kramdown'
require 'kramdown-parser-gfm'

module WikiRender
  KRAMDOWN_OPTS = {
    input: 'GFM',
    auto_ids: true,            # heading id= for bootstrap-toc sidebar + anchors
    hard_wrap: false,          # GFM parser defaults this to true; al-folio uses false
    smart_quotes: %w[lsquo rsquo ldquo rdquo],
    syntax_highlighter: 'rouge',
    syntax_highlighter_opts: {
      css_class: 'highlight',
      default_lang: 'plaintext',
      span: { line_numbers: false },
      block: { line_numbers: false, start_line: 1 }
    }
  }.freeze

  module_function

  # First top-level "# Title" becomes the page title and is stripped from the body.
  def extract_title(md)
    title = nil
    out = []
    md.each_line do |ln|
      if title.nil? && ln =~ /\A\#\s+(.+?)\s*\z/
        title = Regexp.last_match(1).strip
        next
      end
      out << ln
    end
    [title, out.join]
  end

  def protect_code(md, store)
    out = md.gsub(/```[\s\S]*?```|~~~[\s\S]*?~~~/m) do |m|
      k = "zzCODEzz#{store.size}zz"
      store[k] = [:code, m]
      k
    end
    out.gsub(/(`+)(?:(?!\1)[\s\S])*?\1/) do |m|
      k = "zzCODEzz#{store.size}zz"
      store[k] = [:code, m]
      k
    end
  end

  def restore_code(text, store)
    store.each { |k, (t, v)| text = text.gsub(k) { v } if t == :code }
    text
  end

  def protect_math(md, store)
    out = md.gsub(/\$\$[\s\S]*?\$\$/) do |m|
      k = "zzMATHzz#{store.size}zz"
      store[k] = [:disp, m[2...-2]]
      k
    end
    out.gsub(/(?<!\\)\$([^$\n]+?)(?<!\\)\$/) do
      inner = Regexp.last_match(1)
      k = "zzMATHzz#{store.size}zz"
      store[k] = [:inl, inner]
      k
    end
  end

  # Escape only &, <, > so the HTML is well-formed (e.g. $a<b$ must not become a <b> tag).
  # Backslashes are left intact. The browser decodes these back in textContent, which is
  # what MathJax actually parses — this mirrors kramdown's own math-engine output.
  def escape_math(s)
    s.gsub('&', '&amp;').gsub('<', '&lt;').gsub('>', '&gt;')
  end

  def restore_math(html, store)
    store.each do |k, (t, v)|
      next unless %i[disp inl].include?(t)
      e = escape_math(v)
      repl = t == :disp ? "\\[#{e}\\]" : "\\(#{e}\\)"
      html = html.gsub(k) { repl } # block form: replacement is literal (math has many backslashes)
    end
    html
  end

  # Rewrite internal [text](foo.md) / [text](foo.md#anchor) to site permalinks.
  def rewrite_links(md, link_map)
    md.gsub(/\]\(([^)#\s]+)(#[^)]*)?\)/) do
      target = Regexp.last_match(1)
      anchor = Regexp.last_match(2) || ''
      link_map.key?(target) ? "](#{link_map[target]}#{anchor})" : "](#{target}#{anchor})"
    end
  end

  def to_html(md, link_map)
    store = {}
    body = protect_code(md, store)
    body = protect_math(body, store)
    body = rewrite_links(body, link_map)
    body = restore_code(body, store)
    html = Kramdown::Document.new(body, KRAMDOWN_OPTS).to_html
    restore_math(html, store)
  end

  # Full pipeline: returns [title, html]
  def render(md, link_map)
    title, body = extract_title(md)
    [title, to_html(body, link_map)]
  end
end
