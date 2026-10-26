$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  $("a").removeClass("waves-effect waves-light");

  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });
    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    var $wiki = $(".wiki-content");
    if ($wiki.length) {
      // Wiki questions use a third navigation level, beyond bootstrap-toc's default depth.
      $myNav.empty().attr("data-toggle", "toc").attr("aria-label", "本页目录").addClass("wiki-toc");
      var parents = [{ level: 1, $item: $myNav }];
      $wiki.find("h2, h3, h4").not("[data-toc-skip]").each(function () {
        var level = Toc.helpers.getNavLevel(this);
        while (parents[parents.length - 1].level >= level) {
          parents.pop();
        }
        var $parent = parents[parents.length - 1].$item;
        var $list = $parent.children("ul").first();
        if (!$list.length) {
          $list = Toc.helpers.createChildNavList($parent);
        }
        var $item = Toc.helpers.generateNavItem(this);
        $list.append($item);
        parents.push({ level: level, $item: $item });
      });
    } else {
      Toc.init($myNav);
    }
    $("body").scrollspy({
      target: navSelector,
    });
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let jupyterTheme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (jupyterTheme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // trigger popovers
  $('[data-toggle="popover"]').popover({
    trigger: "hover",
  });
});
