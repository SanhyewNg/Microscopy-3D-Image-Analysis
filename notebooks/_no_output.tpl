{%- extends 'full.tpl' -%}

{% block input_group -%}
<div class="input_hidden">
{{ super() }}
</div>
{% endblock input_group %}

{%- block header -%}
{{ super() }}

<style type="text/css">
//div.output_wrapper {
//  margin-top: 0px;
//}

.input_hidden {
  display: none;
//  margin-top: 5px;
}

div.prompt {
    display: none;
}

.CodeMirror{
    font-family: "Consolas", sans-serif;
}

p {font-size:14px;}

</style>

{%- endblock header -%}